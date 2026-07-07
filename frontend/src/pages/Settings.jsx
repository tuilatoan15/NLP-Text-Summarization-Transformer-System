import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings as SettingsIcon, User, Globe, Palette, Lock, Database, Terminal,
  Save, RotateCcw, Check, AlertCircle, Loader2, Cpu
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useSystemConfigQuery } from '../hooks/useApiQueries';

const SettingSection = ({ icon: Icon, title, subtitle, children }) => (
  <motion.div
    initial={{ opacity: 0, y: 8 }}
    animate={{ opacity: 1, y: 0 }}
    className="ui-card p-6 space-y-4"
  >
    <div className="flex items-start gap-3 mb-5">
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: 'var(--accent-muted)' }}
      >
        <Icon className="w-5 h-5" style={{ color: 'var(--accent)' }} />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">{title}</h3>
        {subtitle && <p className="text-sm text-[var(--text-muted)] mt-1">{subtitle}</p>}
      </div>
    </div>
    {children}
  </motion.div>
);

const SettingItem = ({ label, description, children }) => (
  <div className="space-y-2 pb-4 border-b border-[var(--border-subtle)] last:border-0 last:pb-0">
    <label className="block text-sm font-medium text-[var(--text-primary)]">
      {label}
    </label>
    {description && <p className="text-xs text-[var(--text-muted)]">{description}</p>}
    <div>{children}</div>
  </div>
);

const ToggleButton = ({ active, onChange, label }) => (
  <button
    onClick={() => onChange(!active)}
    className={`px-4 py-2 rounded-lg font-medium transition-all ${
      active
        ? 'bg-[var(--accent)] text-white'
        : 'bg-[var(--bg-muted)] text-[var(--text-secondary)]'
    }`}
  >
    {label}
  </button>
);

const DEFAULT_SETTINGS = {
  profile: {
    name: 'AI Document Hub User',
    email: 'user@example.com',
  },
  api: {
    model: 'openai-gpt-4',
    temperature: 0.7,
    maxTokens: 2048,
  },
  privacy: {
    cacheEnabled: true,
    analytics: true,
    autoSave: true,
  },
  advanced: {
    debugMode: false,
    experimentalFeatures: false,
  }
};

const ConfigRow = ({ label, value }) => (
  <div className="flex justify-between items-center py-2 border-b border-[var(--border-subtle)] last:border-0 text-sm">
    <span className="text-[var(--text-muted)]">{label}</span>
    <span className="font-mono text-xs font-semibold text-[var(--text-primary)]">{String(value ?? '—')}</span>
  </div>
);

const Settings = () => {
  const { t, isDark, toggleTheme, locale, setLanguage } = useApp();
  const { data: backendConfig, isLoading: configLoading, error: configError } = useSystemConfigQuery();
  const [saved, setSaved] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

  React.useEffect(() => {
    try {
      const stored = localStorage.getItem('modelSettings');
      if (stored) {
        const parsed = JSON.parse(stored);
        // Deep merge to prevent missing sections/fields from throwing undefined errors
        const merged = {
          profile: { ...DEFAULT_SETTINGS.profile, ...parsed.profile },
          api: { ...DEFAULT_SETTINGS.api, ...parsed.api },
          privacy: { ...DEFAULT_SETTINGS.privacy, ...parsed.privacy },
          advanced: { ...DEFAULT_SETTINGS.advanced, ...parsed.advanced },
        };
        setSettings(merged);
      }
    } catch (e) {
      console.warn('Failed to load settings from localStorage:', e);
    }
  }, []);

  const handleSave = () => {
    try {
      localStorage.setItem('modelSettings', JSON.stringify(settings));
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) {
      console.error('Failed to save settings to localStorage:', e);
    }
  };

  const handleReset = () => {
    try {
      const defaultSettingsClone = JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
      setSettings(defaultSettingsClone);
      localStorage.setItem('modelSettings', JSON.stringify(defaultSettingsClone));

      // Reset global theme to light and language to vie
      toggleTheme('light');
      setLanguage('vie');

      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) {
      console.error('Failed to reset settings in localStorage:', e);
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div>
        <h1 className="ui-heading-1 mb-1 flex items-center gap-2">
          <SettingsIcon className="w-8 h-8" style={{ color: 'var(--accent)' }} />
          {t('settingsTitle')}
        </h1>
        <p className="ui-text text-[var(--text-muted)]">
          {t('settingsSubtitle')}
        </p>
      </div>

      {/* Success Notification */}
      {saved && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="flex items-center gap-2 px-4 py-3 rounded-lg"
          style={{ backgroundColor: 'var(--success-muted)', borderLeft: '4px solid var(--success)' }}
        >
          <Check size={18} style={{ color: 'var(--success)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--success)' }}>
            {t('settingsSaved')}
          </span>
        </motion.div>
      )}

      <div className="space-y-6">
        {/* Backend system config (read-only) */}
        <SettingSection
          icon={Cpu}
          title="Cấu hình Backend (read-only)"
          subtitle="Giá trị từ server — CUDA, FP16, RAG, cache, Redis"
        >
          {configLoading && (
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] py-4">
              <Loader2 className="animate-spin w-4 h-4" />
              Đang tải cấu hình...
            </div>
          )}
          {configError && (
            <div className="flex items-center gap-2 text-sm text-red-500 py-2">
              <AlertCircle size={16} />
              Không tải được /config — {configError.message}
            </div>
          )}
          {backendConfig && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-xs font-bold uppercase text-[var(--text-faint)] mb-2">GPU / Inference</p>
                <ConfigRow label="FP16" value={backendConfig.gpu?.use_fp16} />
                <ConfigRow label="torch.compile" value={backendConfig.gpu?.use_torch_compile ? 'on' : 'off'} />
                <ConfigRow label="Preload models" value={backendConfig.gpu?.preload_models ? 'yes' : 'lazy'} />
                <ConfigRow label="Max output" value={backendConfig.inference?.max_output_length} />
                <ConfigRow label="Batch (train)" value={backendConfig.inference?.train_batch_size} />
              </div>
              <div>
                <p className="text-xs font-bold uppercase text-[var(--text-faint)] mb-2">RAG / Adaptive</p>
                <ConfigRow label="Generator" value={backendConfig.rag?.generator_type} />
                <ConfigRow label="Embedding" value={backendConfig.rag?.embedding_model} />
                <ConfigRow label="Top-K" value={backendConfig.rag?.top_k_default} />
                <ConfigRow label="Compression" value={backendConfig.rag?.context_compression ? 'on' : 'off'} />
                <ConfigRow label="Adaptive context" value={backendConfig.rag?.adaptive_context ? 'on' : 'off'} />
                <ConfigRow label="Redis" value={backendConfig.infra?.redis_configured ? 'configured' : 'off'} />
              </div>
            </div>
          )}
        </SettingSection>

        {/* Profile Settings */}
        <SettingSection
          icon={User}
          title={t('profile')}
          subtitle={t('settingsProfileSub')}
        >
          <SettingItem label={t('settingsFullName')}>
            <input
              type="text"
              value={settings.profile.name}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  profile: { ...settings.profile, name: e.target.value }
                })
              }
              className="ui-input w-full"
              placeholder={t('settingsNamePlaceholder')}
            />
          </SettingItem>
          <SettingItem label={t('settingsEmail')}>
            <input
              type="email"
              value={settings.profile.email}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  profile: { ...settings.profile, email: e.target.value }
                })
              }
              className="ui-input w-full"
              placeholder="your@email.com"
            />
          </SettingItem>
        </SettingSection>

        {/* Language & Theme */}
        <SettingSection
          icon={Palette}
          title={t('settingsDisplayLang')}
          subtitle={t('settingsDisplayLangSub')}
        >
          <SettingItem label={t('themeLight') + ' / ' + t('themeDark')}>
            <div className="flex gap-2">
              <ToggleButton
                active={!isDark}
                onChange={() => isDark && toggleTheme()}
                label="☀️ Light"
              />
              <ToggleButton
                active={isDark}
                onChange={() => !isDark && toggleTheme()}
                label="🌙 Dark"
              />
            </div>
          </SettingItem>
          <SettingItem label={t('settingsLangLabel')} description={t('settingsLangDesc')}>
            <div className="flex gap-2">
              <ToggleButton
                active={locale === 'vie'}
                onChange={() => setLanguage('vie')}
                label="VI - Tiếng Việt"
              />
              <ToggleButton
                active={locale === 'eng'}
                onChange={() => setLanguage('eng')}
                label="EN - English"
              />
            </div>
          </SettingItem>
        </SettingSection>

        {/* API Configuration */}
        <SettingSection
          icon={Terminal}
          title={t('settingsApiTitle')}
          subtitle={t('settingsApiSub')}
        >
          <SettingItem label={t('settingsModelSelect')} description={t('settingsModelDesc')}>
            <select
              value={settings.api.model}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  api: { ...settings.api, model: e.target.value }
                })
              }
              className="ui-input w-full"
            >
              <option value="openai-gpt-4">OpenAI GPT-4</option>
              <option value="openai-gpt-3.5">OpenAI GPT-3.5 Turbo</option>
              <option value="gemini-pro">Google Gemini Pro</option>
              <option value="local-ollama">Ollama (Local)</option>
            </select>
          </SettingItem>
          <SettingItem
            label={t('settingsTemperature')}
            description={t('settingsTemperatureHint')}
          >
            <div className="space-y-2">
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.api.temperature}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    api: { ...settings.api, temperature: parseFloat(e.target.value) }
                  })
                }
                className="w-full"
              />
              <div className="text-xs text-[var(--text-muted)]">
                {t('settingsCurrent')}: {settings.api.temperature.toFixed(1)}
              </div>
            </div>
          </SettingItem>
          <SettingItem
            label={t('settingsMaxLength')}
            description={t('settingsMaxLengthHint')}
          >
            <input
              type="number"
              value={settings.api.maxTokens}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  api: { ...settings.api, maxTokens: parseInt(e.target.value) }
                })
              }
              className="ui-input w-full"
              min="100"
              max="8192"
              step="100"
            />
          </SettingItem>
        </SettingSection>

        {/* Privacy & Data */}
        <SettingSection
          icon={Lock}
          title={t('settingsPrivacyTitle')}
          subtitle={t('settingsPrivacySub')}
        >
          <SettingItem
            label={t('settingsCacheManage')}
            description={t('settingsCacheDesc')}
          >
            <div className="flex gap-3 items-center">
              <input
                type="checkbox"
                checked={settings.privacy.cacheEnabled}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    privacy: { ...settings.privacy, cacheEnabled: e.target.checked }
                  })
                }
                className="w-4 h-4"
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {settings.privacy.cacheEnabled ? t('settingsEnabled') : t('settingsDisabled')}
              </span>
            </div>
          </SettingItem>
          <SettingItem
            label={t('settingsAnalyticsLabel')}
            description={t('settingsAnalyticsDesc')}
          >
            <div className="flex gap-3 items-center">
              <input
                type="checkbox"
                checked={settings.privacy.analytics}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    privacy: { ...settings.privacy, analytics: e.target.checked }
                  })
                }
                className="w-4 h-4"
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {settings.privacy.analytics ? t('settingsEnabled') : t('settingsDisabled')}
              </span>
            </div>
          </SettingItem>
          <SettingItem
            label={t('settingsAutoSaveLabel')}
            description={t('settingsAutoSaveDesc')}
          >
            <div className="flex gap-3 items-center">
              <input
                type="checkbox"
                checked={settings.privacy.autoSave}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    privacy: { ...settings.privacy, autoSave: e.target.checked }
                  })
                }
                className="w-4 h-4"
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {settings.privacy.autoSave ? t('settingsEnabled') : t('settingsDisabled')}
              </span>
            </div>
          </SettingItem>
        </SettingSection>

        {/* Advanced Settings */}
        <SettingSection
          icon={Palette}
          title={t('settingsAdvancedTitle')}
          subtitle={t('settingsAdvancedSub')}
        >
          <SettingItem
            label={t('settingsDebugLabel')}
            description={t('settingsDebugDesc')}
          >
            <div className="flex gap-3 items-center">
              <input
                type="checkbox"
                checked={settings.advanced.debugMode}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    advanced: { ...settings.advanced, debugMode: e.target.checked }
                  })
                }
                className="w-4 h-4"
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {settings.advanced.debugMode ? t('settingsEnabled') : t('settingsDisabled')}
              </span>
            </div>
          </SettingItem>
          <SettingItem
            label={t('settingsExperimentalLabel')}
            description={t('settingsExperimentalDesc')}
          >
            <div className="flex gap-3 items-center">
              <input
                type="checkbox"
                checked={settings.advanced.experimentalFeatures}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    advanced: { ...settings.advanced, experimentalFeatures: e.target.checked }
                  })
                }
                className="w-4 h-4"
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {settings.advanced.experimentalFeatures ? t('settingsEnabled') : t('settingsDisabled')}
              </span>
            </div>
          </SettingItem>
        </SettingSection>
      </div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex gap-3 pt-6 border-t border-[var(--border-subtle)]"
      >
        <button
          onClick={handleSave}
          className="ui-btn-primary flex items-center gap-2 px-6 py-3 rounded-lg font-medium cursor-pointer"
        >
          <Save size={18} />
          {t('settingsSave')}
        </button>
        <button
          onClick={handleReset}
          className="ui-btn-secondary flex items-center gap-2 px-6 py-3 rounded-lg font-medium cursor-pointer"
        >
          <RotateCcw size={18} />
          {t('settingsReset')}
        </button>
      </motion.div>
    </div>
  );
};

export default Settings;
