import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-white text-black">
      <div className="max-w-7xl mx-auto px-4 py-24">
        {/* Hero Section */}
        <div className="text-center space-y-8 mb-24">
          <h1 className="text-6xl sm:text-7xl font-bold tracking-tight text-black">
            InsightCore
          </h1>
          <p className="text-xl sm:text-2xl text-gray-500 max-w-3xl mx-auto font-light leading-relaxed">
            Experience the next generation of AI interaction.
            <br />
            Powerful. Intuitive. Revolutionary.
          </p>
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mt-12">
            <Link
              href="/register"
              className="px-8 py-4 bg-blue-600 text-white rounded-full text-lg font-medium transition-all duration-300 hover:bg-blue-700 w-full sm:w-auto"
            >
              Get Started
            </Link>
            <Link
              href="/login"
              className="px-8 py-4 bg-gray-200 text-gray-800 rounded-full text-lg font-medium transition-all duration-300 hover:bg-gray-300 w-full sm:w-auto"
            >
              Sign In
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-24">
          <div className="text-center">
            <div className="h-20 w-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-6">
              <svg
                className="h-10 w-10 text-blue-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-black mb-4">
              Powerful RAG Engine
            </h3>
            <p className="text-gray-500 leading-relaxed">
              Harness the power of state-of-the-art AI models with our
              advanced retrieval and generation system. Built for performance
              and scalability.
            </p>
          </div>

          <div className="text-center">
            <div className="h-20 w-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-6">
              <svg
                className="h-10 w-10 text-blue-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-black mb-4">
              Seamless Integration
            </h3>
            <p className="text-gray-500 leading-relaxed">
              Effortlessly connect with your existing tech stack. Our flexible
              API and comprehensive SDK make integration a breeze.
            </p>
          </div>

          <div className="text-center">
            <div className="h-20 w-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-6">
              <svg
                className="h-10 w-10 text-blue-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-black mb-4">
              Real-time Analytics
            </h3>
            <p className="text-gray-500 leading-relaxed">
              Gain deep insights into your RAG system's performance with our
              comprehensive analytics dashboard and monitoring tools.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
