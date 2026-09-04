export type Quality = 'fast' | 'balanced' | 'accurate' | 'maximum';
export type ModelName =
  | 'v6-tiny-det'
  | 'v5-mobile-det'
  | 'v6-small-det'
  | 'v6-medium-det'
  | 'v6-tiny-rec'
  | 'v6-small-rec'
  | 'v6-medium-rec'
  | 'v5-latin-rec'
  | 'v5-eslav-rec'
  | 'v5-korean-rec';

export interface Artifact {
  url: string;
  bytes: number;
  sha256: string;
}

export interface ModelEntry {
  label: string;
  shortLabel: string;
  artifacts: Partial<Record<'inference.onnx' | 'inference.yml', Artifact>>;
}

function model(
  label: string,
  shortLabel: string,
  repository: string,
  revision: string,
  modelBytes: number,
  modelSha256: string,
  configBytes?: number,
  configSha256?: string,
): ModelEntry {
  const base = `https://huggingface.co/${repository}/resolve/${revision}`;
  return {
    label,
    shortLabel,
    artifacts: {
      'inference.onnx': { url: `${base}/inference.onnx`, bytes: modelBytes, sha256: modelSha256 },
      ...(configBytes && configSha256
        ? { 'inference.yml': { url: `${base}/inference.yml`, bytes: configBytes, sha256: configSha256 } }
        : {}),
    },
  };
}

export const MODELS: Record<ModelName, ModelEntry> = {
  'v6-tiny-det': model('PP-OCRv6 Tiny detector', 'Tiny detector', 'PaddlePaddle/PP-OCRv6_tiny_det_onnx', '2ba1506c0380b8f0b03dd142459aac66d4421f6c', 1_780_590, '193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8'),
  'v5-mobile-det': model('PP-OCRv5 Mobile detector', 'Mobile detector', 'PaddlePaddle/PP-OCRv5_mobile_det_onnx', 'e6f4fa85f00e168c862bc462aebca69eef9b3d3d', 4_826_518, 'a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d'),
  'v6-small-det': model('PP-OCRv6 Small detector', 'Small detector', 'PaddlePaddle/PP-OCRv6_small_det_onnx', '28fe5895c24fd108c19eb3e8479f4ab385fbfc62', 9_880_512, 'd73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e'),
  'v6-medium-det': model('PP-OCRv6 Medium detector', 'Medium detector', 'PaddlePaddle/PP-OCRv6_medium_det_onnx', '61323801669c338b7891481ec7bac61ce31b576a', 62_032_837, 'eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1'),
  'v6-tiny-rec': model('PP-OCRv6 Tiny recognizer', 'Tiny recognizer', 'PaddlePaddle/PP-OCRv6_tiny_rec_onnx', '2612ab37152ae0a677521bae4e1e3d4fb4cf7c30', 4_462_639, '9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6', 55_571, '66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1'),
  'v6-small-rec': model('PP-OCRv6 Small recognizer', 'Small recognizer', 'PaddlePaddle/PP-OCRv6_small_rec_onnx', 'b8f84f0b80c529de40b4fbb3544b84fa7233a513', 21_159_378, '5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634', 150_579, 'ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1'),
  'v6-medium-rec': model('PP-OCRv6 Medium recognizer', 'Medium recognizer', 'PaddlePaddle/PP-OCRv6_medium_rec_onnx', '50c7eacafc52fa7bcf4194e8cd08e46f8558504b', 76_554_979, '9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba', 150_580, '991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129'),
  'v5-latin-rec': model('PP-OCRv5 Latin recognizer', 'Latin pack', 'PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx', '89d3a50e2c27e2e7cceeab0e944c25c807d5db4f', 8_042_023, '7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498', 6_817, '0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84'),
  'v5-eslav-rec': model('PP-OCRv5 Eastern Slavic recognizer', 'Cyrillic pack', 'PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx', '9a32171fc5718746875e1a261818884517975013', 7_887_627, 'b3018ef2b09a0250b6e0c8e871c927098363e5fd4df890cc68e8358eb0aaf1bd', 4_538, '025039bac23eb4a308efcefa4d58eab3af440767815c6ba6938468bf6353ee5a'),
  'v5-korean-rec': model('PP-OCRv5 Korean recognizer', 'Korean pack', 'PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx', '5c6f574b8e2230adf4287b33e736d71b9fabd28e', 13_418_787, '92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08', 96_039, 'f757fa1c40e99edcf27e9cce879b93eb2a51fa46f5ef39095689b8c37dd75998'),
};

export const QUALITY = {
  fast: {
    label: 'Fast', detectorThreshold: 0.20, boxThreshold: 0.40, unclipRatio: 1.40,
    maxSide: 960, batchSize: 16, widthBudget: 12_288,
  },
  balanced: {
    label: 'Balanced', detectorThreshold: 0.30, boxThreshold: 0.60, unclipRatio: 1.50,
    maxSide: 1280, batchSize: 8, widthBudget: 8_192,
  },
  accurate: {
    label: 'Accurate', detectorThreshold: 0.20, boxThreshold: 0.45, unclipRatio: 1.40,
    maxSide: 1600, batchSize: 8, widthBudget: 8_192,
  },
  maximum: {
    label: 'Maximum', detectorThreshold: 0.20, boxThreshold: 0.45, unclipRatio: 1.40,
    maxSide: 2048, batchSize: 8, widthBudget: 8_192,
  },
} as const;

export type QualityProfile = (typeof QUALITY)[Quality];

export function detectorFor(quality: Quality): ModelName {
  if (quality === 'fast') return 'v6-tiny-det';
  if (quality === 'accurate') return 'v6-small-det';
  if (quality === 'maximum') return 'v6-medium-det';
  return 'v5-mobile-det';
}

export function primaryRecognizerFor(quality: Quality): ModelName {
  if (quality === 'fast') return 'v6-tiny-rec';
  if (quality === 'maximum') return 'v6-medium-rec';
  return 'v6-small-rec';
}

export function modelBytes(name: ModelName): number {
  return Object.values(MODELS[name].artifacts).reduce((sum, artifact) => sum + (artifact?.bytes ?? 0), 0);
}