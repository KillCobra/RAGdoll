import Image from 'next/image'
import AnalysisForm from './components/AnalysisForm'

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-extrabold mb-4 text-gray-900">
            Risk Analysis System
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Leverage AI-powered analysis to identify and assess potential risks for your business expansion
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-xl shadow-2xl p-8 mb-12">
          {/* Instructions */}
          <div className="mb-10 border-b pb-8">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">How it works</h2>
            <ol className="list-decimal pl-6 space-y-4 text-gray-600">
              <li className="pl-2">Enter your company description in detail, including your core business activities, current market position, and goals</li>
              <li className="pl-2">Specify your target market or sector for expansion</li>
              <li className="pl-2">Submit your information for comprehensive analysis</li>
              <li className="pl-2">Receive a detailed risk assessment report with actionable insights</li>
            </ol>
          </div>

          {/* Form Component */}
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-xl font-medium mb-6 text-gray-800">Enter Your Details</h3>
            <AnalysisForm />
          </div>
        </div>

        {/* Footer Section */}
        <footer className="text-center text-gray-600">
          <p className="text-sm mb-4">Powered by Advanced RAG Technology</p>
          <div className="flex justify-center gap-6 items-center">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
              <span>ChromaDB</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              <span>LangChain</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
              <span>Gemini API</span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  )
}