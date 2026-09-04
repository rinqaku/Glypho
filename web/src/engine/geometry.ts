import type { Point, Region } from './types';

export interface HeatmapLike {
  dims: readonly number[];
  data: ArrayLike<number>;
}

export interface DetectionProfile {
  detectorThreshold: number;
  boxThreshold: number;
  unclipRatio: number;
}

interface Rect {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  angle: number;
  points: [Point, Point, Point, Point];
}

const MAX_REGIONS = 1000;
const MIN_SIZE = 3;

/**
 * Browser implementation of the DB quad post-process used by native Glypho.
 *
 * The important order matches Paddle/OAR: threshold -> connected contour ->
 * minimum-area rectangle -> box score -> unclip -> minimum-area rectangle -> scale.
 * We deliberately keep this module dependency-free so it can later be replaced by
 * a small shared Rust/WASM core without changing the worker/UI contracts.
 */
export async function extractRegions(
  output: HeatmapLike,
  sourceWidth: number,
  sourceHeight: number,
  profile: DetectionProfile,
): Promise<Region[]> {
  const height = Number(output.dims.at(-2));
  const width = Number(output.dims.at(-1));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return [];

  const mapSize = width * height;
  const start = Math.max(0, output.data.length - mapSize);
  const values = output.data;
  const visited = new Uint8Array(mapSize);
  const queue = new Int32Array(mapSize);
  const candidates: Region[] = [];
  const threshold = profile.detectorThreshold;

  for (let y0 = 0; y0 < height; y0 += 1) {
    for (let x0 = 0; x0 < width; x0 += 1) {
      const origin = y0 * width + x0;
      if (visited[origin] || Number(values[start + origin]) <= threshold) continue;

      let head = 0;
      let tail = 1;
      queue[0] = origin;
      visited[origin] = 1;
      const boundary: Point[] = [];

      while (head < tail) {
        const index = queue[head++];
        const x = index % width;
        const y = Math.floor(index / width);
        let isBoundary = false;

        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
              isBoundary = true;
              continue;
            }
            const neighbor = ny * width + nx;
            if (Number(values[start + neighbor]) <= threshold) {
              isBoundary = true;
              continue;
            }
            if (!visited[neighbor]) {
              visited[neighbor] = 1;
              queue[tail++] = neighbor;
            }
          }
        }

        if (isBoundary) boundary.push({ x, y });
      }

      if (tail < 4 || boundary.length < 3) continue;
      const hull = convexHull(boundary);
      if (hull.length < 3) continue;

      const mini = minimumAreaRect(hull);
      if (!mini || Math.min(mini.width, mini.height) < MIN_SIZE) continue;

      const score = boxScoreFast(values, start, width, height, mini.points);
      if (score < profile.boxThreshold) continue;

      const expanded = unclipRect(mini, profile.unclipRatio);
      if (Math.min(expanded.width, expanded.height) < MIN_SIZE + 2) continue;

      const region = scaleRect(expanded, width, height, sourceWidth, sourceHeight, score);
      if (region.width >= 4 && region.height >= 4) candidates.push(region);
      if (candidates.length >= MAX_REGIONS) break;
    }
    if (candidates.length >= MAX_REGIONS) break;
    if ((y0 & 63) === 63) await yieldToBrowser();
  }

  return sortReadingOrder(candidates);
}

export function convexHull(points: Point[]): Point[] {
  if (points.length <= 3) return [...points];
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const unique: Point[] = [];
  for (const point of sorted) {
    const last = unique.at(-1);
    if (!last || last.x !== point.x || last.y !== point.y) unique.push(point);
  }
  if (unique.length <= 3) return unique;

  const lower: Point[] = [];
  for (const point of unique) {
    while (lower.length >= 2 && cross(lower.at(-2)!, lower.at(-1)!, point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Point[] = [];
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index];
    while (upper.length >= 2 && cross(upper.at(-2)!, upper.at(-1)!, point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function cross(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

export function minimumAreaRect(points: Point[]): Rect | undefined {
  if (points.length < 3) return undefined;
  let best: { area: number; minU: number; maxU: number; minV: number; maxV: number; angle: number } | undefined;
  // Match OpenCV/OAR more closely: evaluate every hull edge. Subsampling edges
  // changes the min-area angle and can noticeably distort recognition crops.
  for (let index = 0; index < points.length; index += 1) {
    const next = points[(index + 1) % points.length];
    const current = points[index];
    const angle = Math.atan2(next.y - current.y, next.x - current.x);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;

    for (const point of points) {
      const u = point.x * cos + point.y * sin;
      const v = -point.x * sin + point.y * cos;
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }

    const area = (maxU - minU) * (maxV - minV);
    if (!best || area < best.area) best = { area, minU, maxU, minV, maxV, angle };
  }

  if (!best || !Number.isFinite(best.area) || best.area <= 0) return undefined;
  const width = best.maxU - best.minU;
  const height = best.maxV - best.minV;
  const centerU = (best.minU + best.maxU) / 2;
  const centerV = (best.minV + best.maxV) / 2;
  const cos = Math.cos(best.angle);
  const sin = Math.sin(best.angle);
  const centerX = centerU * cos - centerV * sin;
  const centerY = centerU * sin + centerV * cos;
  return rectFromCenter(centerX, centerY, width, height, best.angle);
}

function rectFromCenter(centerX: number, centerY: number, width: number, height: number, angle: number): Rect {
  // Keep the long side horizontal in crop coordinates, mirroring Paddle's mini-box ordering.
  if (height > width) {
    [width, height] = [height, width];
    angle += Math.PI / 2;
  }
  angle = normalizeAngle(angle);
  const halfW = width / 2;
  const halfH = height / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const local: [number, number][] = [
    [-halfW, -halfH],
    [halfW, -halfH],
    [halfW, halfH],
    [-halfW, halfH],
  ];
  const points = local.map(([u, v]) => ({
    x: centerX + u * cos - v * sin,
    y: centerY + u * sin + v * cos,
  })) as [Point, Point, Point, Point];
  return { centerX, centerY, width, height, angle, points };
}

function unclipRect(rect: Rect, ratio: number): Rect {
  // Paddle/OAR use delta = polygon_area * unclip_ratio / perimeter and then
  // recompute a min-area rectangle from the inflated polygon. For a rectangle,
  // expanding each side by delta is the same geometry without bringing Clipper2
  // into the browser bundle.
  const area = rect.width * rect.height;
  const perimeter = 2 * (rect.width + rect.height);
  const delta = perimeter > 0 ? area * ratio / perimeter : 0;
  return rectFromCenter(rect.centerX, rect.centerY, rect.width + 2 * delta, rect.height + 2 * delta, rect.angle);
}

function boxScoreFast(
  values: ArrayLike<number>,
  start: number,
  width: number,
  height: number,
  quad: [Point, Point, Point, Point],
): number {
  const bounds = quadBounds(quad);
  const minX = clamp(Math.floor(bounds.x), 0, width - 1);
  const maxX = clamp(Math.ceil(bounds.x + bounds.width), 0, width - 1);
  const minY = clamp(Math.floor(bounds.y), 0, height - 1);
  const maxY = clamp(Math.ceil(bounds.y + bounds.height), 0, height - 1);
  let sum = 0;
  let count = 0;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (!pointInConvexQuad(x + 0.5, y + 0.5, quad)) continue;
      sum += Number(values[start + y * width + x]);
      count += 1;
    }
  }
  return count ? sum / count : 0;
}

function pointInConvexQuad(x: number, y: number, quad: [Point, Point, Point, Point]): boolean {
  let sign = 0;
  for (let index = 0; index < 4; index += 1) {
    const a = quad[index];
    const b = quad[(index + 1) % 4];
    const value = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    if (Math.abs(value) < 1e-6) continue;
    const current = value > 0 ? 1 : -1;
    if (!sign) sign = current;
    else if (sign !== current) return false;
  }
  return true;
}

function scaleRect(
  rect: Rect,
  mapWidth: number,
  mapHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  score: number,
): Region {
  const scaleX = sourceWidth / mapWidth;
  const scaleY = sourceHeight / mapHeight;
  const quad = rect.points.map((point) => ({
    x: Math.round(clamp(point.x * scaleX, 0, sourceWidth)),
    y: Math.round(clamp(point.y * scaleY, 0, sourceHeight)),
  })) as [Point, Point, Point, Point];
  const bounds = quadBounds(quad);
  const topWidth = distance(quad[0], quad[1]);
  const bottomWidth = distance(quad[3], quad[2]);
  const leftHeight = distance(quad[0], quad[3]);
  const rightHeight = distance(quad[1], quad[2]);
  const cropWidth = Math.max(1, (topWidth + bottomWidth) / 2);
  const cropHeight = Math.max(1, (leftHeight + rightHeight) / 2);
  const angle = normalizeAngle(Math.atan2(quad[1].y - quad[0].y, quad[1].x - quad[0].x));
  return {
    ...bounds,
    quad,
    centerX: quad.reduce((sum, point) => sum + point.x, 0) / 4,
    centerY: quad.reduce((sum, point) => sum + point.y, 0) / 4,
    cropWidth,
    cropHeight,
    angle,
    score,
  };
}

export function sortReadingOrder<T extends { x: number; y: number; width: number; height: number }>(lines: T[]): T[] {
  const sorted = [...lines].sort((left, right) => left.y - right.y || left.x - right.x);
  const rows: T[][] = [];
  for (const line of sorted) {
    const row = rows.at(-1);
    const reference = row?.at(-1);
    const tolerance = reference ? Math.min(line.height, reference.height) * 0.5 : 0;
    const sameRow = reference && Math.abs(centerY(line) - centerY(reference)) <= tolerance;
    if (sameRow) row!.push(line);
    else rows.push([line]);
  }
  for (const row of rows) row.sort((left, right) => left.x - right.x);
  return rows.flat();
}

export function quadBounds(quad: readonly Point[]): { x: number; y: number; width: number; height: number } {
  const xs = quad.map((point) => point.x);
  const ys = quad.map((point) => point.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  const right = Math.max(...xs);
  const bottom = Math.max(...ys);
  return { x, y, width: right - x, height: bottom - y };
}

export function distance(a: Point, b: Point): number { return Math.hypot(b.x - a.x, b.y - a.y); }

export function normalizeAngle(angle: number): number {
  while (angle <= -Math.PI / 2) angle += Math.PI;
  while (angle > Math.PI / 2) angle -= Math.PI;
  return angle;
}

function centerY(region: { y: number; height: number }): number { return region.y + region.height / 2; }
function clamp(value: number, min: number, max: number): number { return Math.min(max, Math.max(min, value)); }
function yieldToBrowser(): Promise<void> { return new Promise((resolve) => setTimeout(resolve, 0)); }