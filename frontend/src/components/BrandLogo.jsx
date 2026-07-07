import React, { memo } from 'react';
import { Link } from 'react-router-dom';

/** Document stack + AI sparkle motif for the app brand mark */
const HubLogoIcon = ({ className = 'w-5 h-5' }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    className={className}
    aria-hidden="true"
  >
    <rect x="8" y="2.5" width="11" height="14.5" rx="1.5" fill="white" fillOpacity="0.22" />
    <path
      d="M5.5 5.5h8.75L17 8.25v11a1.5 1.5 0 0 1-1.5 1.5H5.5A1.5 1.5 0 0 1 4 19.25V7A1.5 1.5 0 0 1 5.5 5.5Z"
      fill="white"
      fillOpacity="0.95"
    />
    <path
      d="M14.25 5.5V8.25H17"
      stroke="white"
      strokeOpacity="0.45"
      strokeWidth="0.9"
      strokeLinejoin="round"
    />
    <path
      d="M7.5 10.5h6M7.5 13h4.75M7.5 15.5h3.5"
      stroke="#0c4a6e"
      strokeOpacity="0.38"
      strokeWidth="1.1"
      strokeLinecap="round"
    />
    <path
      d="M18.25 3.75l.38 1.14 1.14.38-1.14.38-.38 1.14-.38-1.14-1.14-.38 1.14-.38.38-1.14z"
      fill="white"
    />
    <circle cx="19.75" cy="6.75" r="0.55" fill="white" fillOpacity="0.75" />
  </svg>
);

const SIZE = {
  sm: { box: 'w-8 h-8', icon: 'w-4 h-4', radius: 'rounded-lg' },
  md: { box: 'w-9 h-9', icon: 'w-5 h-5', radius: 'rounded-xl' },
  lg: { box: 'w-10 h-10', icon: 'w-6 h-6', radius: 'rounded-xl' },
};

const BrandLogo = ({ size = 'md', className = '' }) => {
  const { box, icon, radius } = SIZE[size] ?? SIZE.md;

  return (
    <Link
      to="/"
      title="AI Document Hub — Dashboard"
      aria-label="Go to Dashboard"
      className={[
        box,
        radius,
        'relative flex items-center justify-center shrink-0',
        'bg-gradient-to-br from-sky-400 via-sky-500 to-blue-700',
        'shadow-md shadow-sky-600/30',
        'ring-1 ring-inset ring-white/25',
        'transition-transform duration-200 ease-out',
        'hover:scale-105 active:scale-95',
        'cursor-pointer',
        'before:absolute before:inset-0 before:rounded-[inherit]',
        'before:bg-gradient-to-t before:from-black/10 before:to-white/15 before:pointer-events-none',
        className,
      ].join(' ')}
    >
      <HubLogoIcon className={`${icon} relative z-[1] drop-shadow-sm`} />
    </Link>
  );
};

export default memo(BrandLogo);
