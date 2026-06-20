export interface ModelSettings {
  abstractiveModel: string;
  extractiveAlgorithm: string;
  temperature: number;
  maxLength: number;
  beamSearch: number;
  repetitionPenalty: number;
  extractiveSentences: number;
  similarityThreshold: number;
}
