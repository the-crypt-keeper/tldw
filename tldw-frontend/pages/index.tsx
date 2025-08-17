import { Layout } from '@/components/layout/Layout';
import { Button } from '@/components/ui/Button';
import Link from 'next/link';
import { useAuth } from '@/hooks/useAuth';

export default function Home() {
  const { isAuthenticated, user } = useAuth();
  
  return (
    <Layout>
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
          TLDW Server
        </h1>
        <p className="mt-6 text-lg leading-8 text-gray-600">
          Too Long; Didn't Watch - Your personal media research assistant
        </p>
        
        <div className="mt-10 flex items-center justify-center gap-x-6">
          {isAuthenticated ? (
            <>
              <Link href="/media">
                <Button size="lg">Browse Media</Button>
              </Link>
              <Link href="/chat">
                <Button size="lg" variant="secondary">
                  Start Chat
                </Button>
              </Link>
            </>
          ) : (
            <>
              <Link href="/login">
                <Button size="lg">Get Started</Button>
              </Link>
              <Link href="/about">
                <Button size="lg" variant="ghost">
                  Learn More
                </Button>
              </Link>
            </>
          )}
        </div>
        
        {isAuthenticated && (
          <div className="mt-12">
            <p className="text-sm text-gray-600">
              Welcome back, <span className="font-semibold">{user?.username}</span>!
            </p>
          </div>
        )}
        
        <div className="mt-20 grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="rounded-lg bg-white p-6 shadow">
            <h3 className="text-lg font-semibold text-gray-900">Media Processing</h3>
            <p className="mt-2 text-sm text-gray-600">
              Upload and process videos, audio, PDFs, and more with automatic transcription and analysis.
            </p>
          </div>
          
          <div className="rounded-lg bg-white p-6 shadow">
            <h3 className="text-lg font-semibold text-gray-900">AI-Powered Chat</h3>
            <p className="mt-2 text-sm text-gray-600">
              Chat with your content using advanced AI models with RAG for accurate, contextual responses.
            </p>
          </div>
          
          <div className="rounded-lg bg-white p-6 shadow">
            <h3 className="text-lg font-semibold text-gray-900">Smart Search</h3>
            <p className="mt-2 text-sm text-gray-600">
              Search across all your content with full-text and vector search capabilities.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}