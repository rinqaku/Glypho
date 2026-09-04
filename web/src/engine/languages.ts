import type { Quality } from './models';

export const CYRILLIC_LANGUAGES = new Set(['be', 'ru', 'uk']);
export const CJK_LANGUAGES = new Set(['ja', 'zh']);
export const KOREAN_LANGUAGES = new Set(['ko']);
export const LATIN_LANGUAGES = new Set([
  'af', 'az', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'en', 'es', 'et', 'eu', 'fi', 'fr', 'ga', 'gl',
  'hr', 'hu', 'id', 'is', 'it', 'jv', 'ku', 'la', 'lb', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl', 'no',
  'oc', 'pi', 'pl', 'pt', 'qu', 'rm', 'ro', 'sk', 'sl', 'sq', 'sr-latn', 'sv', 'sw', 'tl', 'tr',
  'uz', 'vi',
]);

export const SUPPORTED_LANGUAGES = new Set([
  ...LATIN_LANGUAGES,
  ...CYRILLIC_LANGUAGES,
  ...CJK_LANGUAGES,
  ...KOREAN_LANGUAGES,
]);

export interface RecognizerPlan {
  primary: boolean;
  latin: boolean;
  cyrillic: boolean;
  korean: boolean;
}

export function normalizeLanguage(value: string): string {
  const language = String(value).trim().toLowerCase().replaceAll('_', '-');
  const aliases: Record<string, string> = {
    bel: 'be', cat: 'ca', ces: 'cs', cze: 'cs', dan: 'da', deu: 'de', ger: 'de', german: 'de',
    dut: 'nl', nld: 'nl', eng: 'en', est: 'et', eus: 'eu', baq: 'eu', fin: 'fi', fra: 'fr', fre: 'fr', french: 'fr',
    glg: 'gl', hrv: 'hr', hun: 'hu', ind: 'id', isl: 'is', ice: 'is', ita: 'it', jpn: 'ja', kor: 'ko', korean: 'ko',
    lav: 'lv', lit: 'lt', nno: 'no', nob: 'no', nor: 'no', pol: 'pl', por: 'pt', ron: 'ro', rum: 'ro', rus: 'ru',
    slk: 'sk', slo: 'sk', slv: 'sl', spa: 'es', 'rs-latin': 'sr-latn', swe: 'sv', tur: 'tr', ukr: 'uk', vie: 'vi',
    'chi-sim': 'zh', 'chi-sim-vert': 'zh', zho: 'zh',
  };
  if (aliases[language]) return aliases[language];
  if (language === 'sr-latn') return language;
  return language.split('-')[0];
}

export function normalizeLanguages(input: string | string[] | undefined | null): string[] {
  const values = Array.isArray(input) ? input : String(input ?? '').split(/[\s,;]+/);
  const result: string[] = [];
  const unsupported: string[] = [];
  for (const value of values) {
    if (!String(value).trim()) continue;
    const normalized = normalizeLanguage(value);
    if (!SUPPORTED_LANGUAGES.has(normalized)) unsupported.push(String(value));
    else if (!result.includes(normalized)) result.push(normalized);
  }
  if (unsupported.length) throw new Error(`Unsupported language: ${unsupported.join(', ')}`);
  return result;
}

export function validateProfileLanguages(quality: Quality, languages: string[]): void {
  if (quality === 'fast' && languages.includes('ja')) {
    throw new Error('Japanese is not available in the fast profile. Use balanced, accurate, or maximum.');
  }
}

export function recognizerPlan(quality: Quality, input: string[] | string): RecognizerPlan {
  const languages = normalizeLanguages(input);
  validateProfileLanguages(quality, languages);
  if (languages.length === 0) {
    return { primary: true, latin: true, cyrillic: true, korean: true };
  }
  const cyrillic = languages.some((language) => CYRILLIC_LANGUAGES.has(language));
  const cjk = languages.some((language) => CJK_LANGUAGES.has(language));
  const korean = languages.some((language) => KOREAN_LANGUAGES.has(language));
  const latinGroup = languages.some((language) => LATIN_LANGUAGES.has(language));
  const latin = languages.some((language) => language !== 'en' && LATIN_LANGUAGES.has(language));
  const scriptGroups = [cyrillic, cjk, korean, latinGroup].filter(Boolean).length;
  return {
    primary: cjk || scriptGroups > 1 || (latinGroup && !latin),
    latin,
    cyrillic,
    korean,
  };
}

export function containsCyrillic(text: string): boolean { return /[\u0400-\u052f]/u.test(text); }
export function containsLatin(text: string): boolean { return /[A-Za-z\u00c0-\u024f\u1e00-\u1eff]/u.test(text); }
export function containsKorean(text: string): boolean { return /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/u.test(text); }
export function containsKana(text: string): boolean { return /[\u3040-\u30ff]/u.test(text); }
export function containsHan(text: string): boolean { return /[\u3400-\u9fff]/u.test(text); }

export type ScriptKind = 'latin' | 'cyrillic' | 'korean';

export function containsScript(text: string, script: ScriptKind): boolean {
  if (script === 'cyrillic') return containsCyrillic(text);
  if (script === 'korean') return containsKorean(text);
  return containsLatin(text);
}

export function scriptTag(text: string): string | undefined {
  if (containsCyrillic(text)) return 'Cyrl';
  if (containsKorean(text)) return 'Kore';
  if (containsKana(text)) return 'Jpan';
  if (containsHan(text)) return 'Hani';
  if (containsLatin(text)) return 'Latn';
  return undefined;
}

export function lineLanguage(text: string, languages: string[]): string | undefined {
  const script = scriptTag(text);
  const source = languages.length ? languages : [...SUPPORTED_LANGUAGES];
  let compatible: string[] = [];
  if (script === 'Cyrl') compatible = source.filter((language) => CYRILLIC_LANGUAGES.has(language));
  else if (script === 'Jpan') compatible = source.filter((language) => language === 'ja');
  else if (script === 'Hani') compatible = source.filter((language) => CJK_LANGUAGES.has(language));
  else if (script === 'Kore') compatible = source.filter((language) => language === 'ko');
  else if (script === 'Latn') compatible = source.filter((language) => LATIN_LANGUAGES.has(language));
  return compatible.length === 1 ? compatible[0] : undefined;
}