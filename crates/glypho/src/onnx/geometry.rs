use super::*;

pub(super) fn crop_ratio(image: &RgbImage) -> f32 {
    image.width() as f32 / image.height().max(1) as f32
}

pub(super) fn should_tile_image(width: u32, height: u32, tile_size: u32) -> bool {
    let long_side = width.max(height);
    let short_side = width.min(height).max(1);
    long_side > tile_size.saturating_mul(3) / 2 && long_side as f32 / short_side as f32 >= 2.0
}

pub(super) fn tile_offsets(length: u32, tile_size: u32) -> Vec<u32> {
    if length <= tile_size {
        return vec![0];
    }
    let span = length - tile_size;
    let intervals = span.div_ceil(tile_size).max(1);
    (0..=intervals)
        .map(|index| ((u64::from(span) * u64::from(index)) / u64::from(intervals)) as u32)
        .collect()
}

pub(super) fn deduplicate_boxes(mut boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
    boxes.sort_by(|left, right| box_area(right).total_cmp(&box_area(left)));
    let mut unique = Vec::with_capacity(boxes.len());
    for candidate in boxes {
        if unique
            .iter()
            .any(|existing| overlap_over_smaller(&candidate, existing) >= 0.65)
        {
            continue;
        }
        unique.push(candidate);
    }
    sort_quad_boxes(&unique)
}

pub(super) fn box_area(box_: &BoundingBox) -> f32 {
    let Some((min_x, min_y, max_x, max_y)) = box_bounds(box_) else {
        return 0.0;
    };
    (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0)
}

pub(super) fn overlap_over_smaller(left: &BoundingBox, right: &BoundingBox) -> f32 {
    let Some((left_x1, left_y1, left_x2, left_y2)) = box_bounds(left) else {
        return 0.0;
    };
    let Some((right_x1, right_y1, right_x2, right_y2)) = box_bounds(right) else {
        return 0.0;
    };
    let intersection_width = (left_x2.min(right_x2) - left_x1.max(right_x1)).max(0.0);
    let intersection_height = (left_y2.min(right_y2) - left_y1.max(right_y1)).max(0.0);
    let smaller = box_area(left).min(box_area(right));
    if smaller <= f32::EPSILON {
        0.0
    } else {
        intersection_width * intersection_height / smaller
    }
}

pub(super) fn box_bounds(box_: &BoundingBox) -> Option<(f32, f32, f32, f32)> {
    let first = box_.points.first()?;
    let mut min_x = first.x;
    let mut min_y = first.y;
    let mut max_x = first.x;
    let mut max_y = first.y;
    for point in &box_.points[1..] {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }
    Some((min_x, min_y, max_x, max_y))
}

pub(super) fn wants_language_group(languages: &[String], group: &[&str]) -> bool {
    languages
        .iter()
        .any(|language| group.contains(&language.as_str()))
}

pub(super) fn wants_specialized_latin(languages: &[String]) -> bool {
    languages
        .iter()
        .any(|language| language != "en" && LATIN_LANGUAGES.contains(&language.as_str()))
}

pub(super) fn quad_from_box(box_: &BoundingBox, width: u32, height: u32) -> Option<Quad> {
    if box_.points.len() >= 4 {
        let points = std::array::from_fn(|index| Point {
            x: box_.points[index].x.clamp(0.0, width as f32),
            y: box_.points[index].y.clamp(0.0, height as f32),
        });
        return Some(Quad { points });
    }
    if box_.points.is_empty() {
        return None;
    }
    let min_x = box_
        .points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, width as f32);
    let min_y = box_
        .points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, height as f32);
    let max_x = box_
        .points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .clamp(0.0, width as f32);
    let max_y = box_
        .points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .clamp(0.0, height as f32);
    (max_x > min_x && max_y > min_y)
        .then(|| Quad::from_rect(min_x, min_y, max_x - min_x, max_y - min_y))
}

pub(super) fn candidate_word_boxes(
    line_box: &BoundingBox,
    candidate: &RecognizedCandidate,
) -> Vec<BoundingBox> {
    let characters = candidate.text.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return Vec::new();
    }
    let centers = if candidate.char_columns.len() == characters.len()
        && candidate.sequence_length > 0
        && candidate.batch_max_ratio > f32::EPSILON
    {
        // Undo dynamic-batch padding before projecting decoder columns onto the crop.
        let effective_columns =
            candidate.sequence_length as f32 * (candidate.crop_ratio / candidate.batch_max_ratio);
        if effective_columns <= f32::EPSILON {
            return Vec::new();
        }
        candidate
            .char_columns
            .iter()
            .map(|column| ((*column as f32 + 0.5) / effective_columns).clamp(0.0, 1.0))
            .collect::<Vec<_>>()
    } else if candidate.char_positions.len() == characters.len() {
        candidate
            .char_positions
            .iter()
            .map(|position| position.clamp(0.0, 1.0))
            .collect::<Vec<_>>()
    } else {
        return Vec::new();
    };

    let mut word_boxes = Vec::new();
    let mut word_start = None;
    for (index, character) in characters.iter().enumerate() {
        if character.is_whitespace() {
            if let Some(start) = word_start.take() {
                word_boxes.push(character_range_box(line_box, &centers, start, index - 1));
            }
        } else if word_start.is_none() {
            word_start = Some(index);
        }
    }
    if let Some(start) = word_start {
        word_boxes.push(character_range_box(
            line_box,
            &centers,
            start,
            characters.len() - 1,
        ));
    }
    word_boxes
        .into_iter()
        .filter(|box_| box_area(box_) > f32::EPSILON)
        .collect()
}

pub(super) fn character_range_box(
    line_box: &BoundingBox,
    centers: &[f32],
    start: usize,
    end: usize,
) -> BoundingBox {
    let left = if start == 0 {
        0.0
    } else {
        (centers[start - 1] + centers[start]) / 2.0
    };
    let right = if end + 1 >= centers.len() {
        1.0
    } else {
        (centers[end] + centers[end + 1]) / 2.0
    };
    slice_line_box(line_box, left.clamp(0.0, 1.0), right.clamp(0.0, 1.0))
}

pub(super) fn slice_line_box(line_box: &BoundingBox, left: f32, right: f32) -> BoundingBox {
    if line_box.points.len() < 4 {
        let Some((min_x, min_y, max_x, max_y)) = box_bounds(line_box) else {
            return BoundingBox::new(Vec::new());
        };
        let width = max_x - min_x;
        return BoundingBox::from_coords(min_x + left * width, min_y, min_x + right * width, max_y);
    }
    let top_left = interpolate_point(&line_box.points[0], &line_box.points[1], left);
    let top_right = interpolate_point(&line_box.points[0], &line_box.points[1], right);
    let bottom_right = interpolate_point(&line_box.points[3], &line_box.points[2], right);
    let bottom_left = interpolate_point(&line_box.points[3], &line_box.points[2], left);
    BoundingBox::new(vec![top_left, top_right, bottom_right, bottom_left])
}

pub(super) fn interpolate_point(left: &OarPoint, right: &OarPoint, position: f32) -> OarPoint {
    OarPoint::new(
        left.x + (right.x - left.x) * position,
        left.y + (right.y - left.y) * position,
    )
}

pub(super) fn build_words(
    line_id: &str,
    region: &OarTextRegion,
    text: &str,
    confidence: f32,
    width: u32,
    height: u32,
) -> Vec<TextWord> {
    let texts = text.split_whitespace().collect::<Vec<_>>();
    let Some(boxes) = region.word_boxes.as_ref() else {
        return Vec::new();
    };
    if boxes.len() != texts.len() {
        return Vec::new();
    }
    boxes
        .iter()
        .zip(texts)
        .enumerate()
        .filter_map(|(index, (box_, text))| {
            let quad = quad_from_box(box_, width, height)?;
            if quad_area(&quad) <= f32::EPSILON {
                return None;
            }
            Some(TextWord {
                id: format!("{line_id}-word-{:04}", index + 1),
                quad,
                text: text.to_owned(),
                confidence: Some(confidence.clamp(0.0, 1.0)),
            })
        })
        .collect()
}

pub(super) fn quad_area(quad: &Quad) -> f32 {
    quad.points
        .iter()
        .zip(quad.points.iter().cycle().skip(1))
        .take(quad.points.len())
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f32>()
        .abs()
        * 0.5
}
