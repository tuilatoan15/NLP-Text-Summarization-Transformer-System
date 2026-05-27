/**
 * Utility functions for model settings calculations and formatting
 */

import type { ModelSettings } from '../types/modelSettings';

/**
 * Estimate reading time for a given word count
 * Average reading speed: 200 words per minute
 */
export const estimateReadingTime = (wordCount: number): string => {
  const readingSpeed = 200;
  const minutes = Math.ceil(wordCount / readingSpeed);
  return `~${minutes} phút`;
};

/**
 * Estimate token count (rough approximation for Vietnamese)
 * Vietnamese: ~1 token per 0.8-1 word
 */
export const estimateTokenCount = (wordCount: number): number => {
  return Math.ceil(wordCount / 0.8);
};

/**
 * Get processing time estimate based on beam search and model
 */
export const estimateProcessingTime = (
  beamSearch: number,
  model: string
): string => {
  let baseTime = 1;

  // Model speed factor
  if (model === 'vit5') baseTime = 1;
  else if (model === 'mt5') baseTime = 1.5;
  else if (model === 'bartpho') baseTime = 2;

  // Beam search factor
  const timeWithBeam = baseTime * (1 + (beamSearch - 1) * 0.3);

  if (timeWithBeam < 2) return '~1-2 giây';
  if (timeWithBeam < 5) return '~3-5 giây';
  if (timeWithBeam < 10) return '~8-12 giây';
  return '~15+ giây';
};

/**
 * Validate settings against constraints
 */
export const validateSettings = (settings: Partial<ModelSettings>): boolean => {
  if (settings.temperature !== undefined) {
    if (settings.temperature < 0 || settings.temperature > 2) return false;
  }

  if (settings.maxLength !== undefined) {
    if (settings.maxLength < 50 || settings.maxLength > 300) return false;
  }

  if (settings.beamSearch !== undefined) {
    if (settings.beamSearch < 1 || settings.beamSearch > 8) return false;
  }

  if (settings.repetitionPenalty !== undefined) {
    if (settings.repetitionPenalty < 1 || settings.repetitionPenalty > 2) return false;
  }

  if (settings.extractiveSentences !== undefined) {
    if (settings.extractiveSentences < 1 || settings.extractiveSentences > 10) return false;
  }

  if (settings.similarityThreshold !== undefined) {
    if (settings.similarityThreshold < 0 || settings.similarityThreshold > 1) return false;
  }

  return true;
};

/**
 * Get quality description based on settings
 */
export const getQualityDescription = (settings: ModelSettings): {
  level: 'Low' | 'Medium' | 'High';
  description: string;
} => {
  const score =
    (settings.beamSearch / 8) * 40 +
    (settings.temperature < 0.5 ? 30 : settings.temperature > 1.2 ? 10 : 20) +
    ((settings.maxLength - 50) / 250) * 30;

  if (score < 35) {
    return {
      level: 'Low',
      description: 'Nhanh nhưng chất lượng có thể bị ảnh hưởng',
    };
  }

  if (score < 65) {
    return {
      level: 'Medium',
      description: 'Cân bằng giữa tốc độ và chất lượng',
    };
  }

  return {
    level: 'High',
    description: 'Chất lượng cao nhưng xử lý chậm hơn',
  };
};

/**
 * Get speed estimate based on settings
 */
export const getSpeedEstimate = (beamSearch: number, model: string): string => {
  let speed: 'Rất nhanh' | 'Nhanh' | 'Trung bình' | 'Chậm' | 'Rất chậm';

  if (beamSearch <= 2) {
    speed = 'Rất nhanh';
  } else if (beamSearch <= 4) {
    speed = 'Nhanh';
  } else if (beamSearch <= 6) {
    speed = 'Trung bình';
  } else {
    speed = 'Chậm';
  }

  // Adjust based on model
  if (model === 'bartpho' && beamSearch > 5) {
    speed = 'Rất chậm';
  }

  return speed;
};

/**
 * Export settings to JSON for backup/sharing
 */
export const exportSettings = (settings: ModelSettings): string => {
  return JSON.stringify(settings, null, 2);
};

/**
 * Import settings from JSON
 */
export const importSettings = (json: string): Partial<ModelSettings> | null => {
  try {
    const parsed = JSON.parse(json);
    if (validateSettings(parsed)) {
      return parsed;
    }
  } catch (error) {
    console.error('Failed to import settings:', error);
  }
  return null;
};

/**
 * Get recommendations for optimal settings based on use case
 */
export const getRecommendations = (useCase: 'speed' | 'quality' | 'balanced') => {
  const recommendations = {
    speed: {
      temperature: 0.3,
      maxLength: 80,
      beamSearch: 1,
      repetitionPenalty: 1.0,
      description: 'Tối ưu cho xử lý nhanh',
    },
    balanced: {
      temperature: 0.7,
      maxLength: 150,
      beamSearch: 4,
      repetitionPenalty: 1.2,
      description: 'Cân bằng tốc độ và chất lượng',
    },
    quality: {
      temperature: 0.9,
      maxLength: 250,
      beamSearch: 8,
      repetitionPenalty: 1.5,
      description: 'Tối ưu cho chất lượng cao',
    },
  };

  return recommendations[useCase];
};
