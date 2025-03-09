import Image from 'next/image'
// import AnalysisForm from './components/AnalysisForm'

export default function Home() {
  return (
    <main className="min-h-screen p-8 bg-black text-white">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-4">
            Web-Enhanced Risk Advisory System
          </h1>
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          <form className="space-y-6">
            {/* Scenario Description */}
            <div>
              <label className="block mb-2">
                Scenario Description:
              </label>
              <textarea 
                className="w-full h-24 p-2 bg-black border border-gray-600 text-white"
                placeholder="Enter your scenario description..."
                required
              />
            </div>

            {/* Specific Questions */}
            <div>
              <label className="block mb-2">
                Specific Questions:
              </label>
              <textarea 
                className="w-full h-24 p-2 bg-black border border-gray-600 text-white"
                placeholder="Enter your specific questions..."
                required
              />
            </div>

            {/* Upload Document */}
            <div>
              <label className="block mb-2">
                Upload Document (Optional):
              </label>
              <input
                type="file"
                className="block w-full text-white"
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full bg-gray-700 text-white py-2 px-4 rounded hover:bg-gray-600"
            >
              Submit Scenario
            </button>
          </form>

          {/* Footer Section */}
          <div className="mt-8">
            <p>Powered by RAG Technology</p>
            <div className="flex gap-2 mt-2">
              <span>• ChromaDB</span>
              <span>• LangChain</span>
              <span>• Gemini API</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}

function AnalysisForm() {
  return (
    <form className="space-y-8">
      {/* Scenario Description */}
      <div>
        <label className="block text-gray-700 font-medium mb-2">
          Scenario Description:
        </label>
        <textarea 
          className="w-full h-40 p-3 border rounded-lg"
          placeholder="Enter your scenario description..."
          required
        />
      </div>

      {/* Specific Questions */}
      <div>
        <label className="block text-gray-700 font-medium mb-2">
          Specific Questions:
        </label>
        <textarea 
          className="w-full h-40 p-3 border rounded-lg"
          placeholder="Enter your specific questions..."
          required
        />
      </div>

      {/* Upload Document */}
      <div>
        <label className="block text-gray-700 font-medium mb-2">
          Upload Document (Optional):
        </label>
        <input
          type="file"
          className="block w-full text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:bg-gray-100 hover:file:bg-gray-200"
        />
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        className="w-full bg-red-600 text-white py-3 rounded-lg hover:bg-red-700 transition-colors"
      >
        Submit Scenario
      </button>
    </form>
  )
}