import styles from './page.module.css';
import AnalysisForm from './components/AnalysisForm';
import BackgroundAnimation from './components/BackgroundAnimation';

export default function Home() {
  return (
    <>
    <BackgroundAnimation />
    <div className={styles.wrapper}>
      <main className={styles.page}>
        <div className={styles.container}>
          {/* Header Section */}
          <header className={styles.header}>
            <h1 className={styles.title}>
              Web-Enhanced Risk Advisory System
            </h1>
            <p className={styles.subtitle}>
              Leverage AI-powered analysis to identify and assess potential risks for your business expansion.
            </p>
          </header>

          {/* Main Content */}
          <section className={styles.mainContent}>
            {/* Instructions */}
            <div className={styles.instructions}>
              <h2 className={styles.sectionTitle}>How it works</h2>
              <ul className={styles.stepsList}>
                <li>Enter your company description in detail, including your core business activities, current market position, and goals.</li>
                <li>Specify your target market or sector for expansion.</li>
                <li>Submit your information for comprehensive analysis.</li>
                <li>Receive a detailed risk assessment report with actionable insights.</li>
              </ul>
            </div>

            {/* Form Component */}
            <div className={styles.formContainer}>
              <h3 className={styles.formTitle}>Enter Your Details</h3>
              <AnalysisForm />
            </div>
          </section>

          {/* Footer Section */}
          <footer className={styles.footer}>
            <p className={styles.footerText}>Powered by Advanced RAG Technology</p>
            <div className={styles.techStack}>
              {[
                { color: styles.blue, label: 'ChromaDB' },
                { color: styles.green, label: 'LangChain' },
                { color: styles.purple, label: 'Gemini API' },
              ].map((tech, index) => (
                <div key={index} className={styles.techItem}>
                  <span className={`${styles.dot} ${tech.color}`}></span>
                  <span>{tech.label}</span>
                </div>
              ))}
            </div>
          </footer>
        </div>
      </main>
    </div>
    </>
  );
}