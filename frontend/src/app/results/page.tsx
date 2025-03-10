'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import styles from './results.module.css';

interface AnalysisResult {
  text: string;
  sources: string[];
}

export default function ResultsPage() {
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const stored = sessionStorage.getItem('analysisResults');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setResults({
          text: parsed.text || parsed.message || '',
          sources: Array.isArray(parsed.sources) ? parsed.sources : []
        });
      } catch (e) {
        setError('Failed to parse results');
        console.error('Error parsing results:', e);
      }
    }
  }, []);

  const formatContent = (text: string) => {
    // Split content into lines
    const lines = text.split('\n');
    let formattedContent: React.ReactNode[] = [];
    let currentList: React.ReactNode[] = [];
    let inList = false;

    lines.forEach((line, index) => {
      // Handle titles (##)
      if (line.startsWith('##')) {
        formattedContent.push(
          <h1 key={`title-${index}`} className={styles.mainTitle}>
            {line.replace('##', '').trim()}
          </h1>
        );
        return;
      }

      // Handle bold text (**text**)
      const boldPattern = /\*\*(.*?)\*\*/g;
      const lineWithBold = line.replace(boldPattern, (match, text) => {
        return `<strong>${text}</strong>`;
      });

      // Handle bullet points
      if (line.trim().startsWith('*')) {
        if (!inList) {
          inList = true;
          currentList = [];
        }
        currentList.push(
          <li key={`list-item-${index}`} 
              className={styles.listItem}
              dangerouslySetInnerHTML={{ __html: lineWithBold.replace('*', '').trim() }} />
        );
      } else {
        if (inList) {
          formattedContent.push(
            <ul key={`list-${index}`} className={styles.list}>
              {currentList}
            </ul>
          );
          inList = false;
          currentList = [];
        }
        
        if (line.trim()) {
          formattedContent.push(
            <p key={`text-${index}`} 
               className={styles.paragraph}
               dangerouslySetInnerHTML={{ __html: lineWithBold }} />
          );
        }
      }
    });

    // Add any remaining list items
    if (inList && currentList.length > 0) {
      formattedContent.push(
        <ul key="final-list" className={styles.list}>
          {currentList}
        </ul>
      );
    }

    return formattedContent;
  };

  if (error) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.container}>
          <div className={styles.error}>
            <p>{error}</p>
            <button onClick={() => router.push('/')} className={styles.button}>
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.container}>
          <div className={styles.loading}>
            <div className="h-4 bg-gray-700 rounded w-3/4 mb-4"></div>
            <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.container}>
        <div className={styles.mainContent}>
          <div className={styles.analysisText}>
            {formatContent(results.text)}
          </div>
          
          {results.sources && results.sources.length > 0 && (
            <>
              <h3 className={styles.sourcesTitle}>Sources</h3>
              <ul className={styles.sourcesList}>
                {results.sources.map((source, index) => (
                  <li key={index} className={styles.sourceItem}>{source}</li>
                ))}
              </ul>
            </>
          )}
          
          <button onClick={() => router.push('/')} className={styles.button}>
            New Analysis
          </button>
        </div>
      </div>
    </div>
  );
}