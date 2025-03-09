interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
  text?: string;
}

export default function LoadingSpinner({ 
  size = 'medium', 
  color = 'blue',
  text = 'Processing...'
}: LoadingSpinnerProps) {
  const sizeClasses = {
    small: 'h-4 w-4',
    medium: 'h-8 w-8',
    large: 'h-12 w-12'
  };

  const colorClasses = {
    blue: 'border-blue-500',
    gray: 'border-gray-500',
    green: 'border-green-500'
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <div 
        className={`
          animate-spin rounded-full
          border-2 border-t-transparent
          ${sizeClasses[size]}
          ${colorClasses[color as keyof typeof colorClasses]}
        `}
      />
      {text && <span className="mt-2 text-sm text-gray-600">{text}</span>}
    </div>
  );
}
