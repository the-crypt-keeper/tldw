import Link from 'next/link';
import { useRouter } from 'next/router';
import { authService } from '@/lib/auth';
import { cn } from '@/lib/utils';

export function Header() {
  const router = useRouter();
  const isAuthenticated = authService.isAuthenticated();
  const user = authService.getUser();
  
  const handleLogout = () => {
    authService.logout();
  };
  
  const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/media', label: 'Media' },
    { href: '/chat', label: 'Chat' },
    { href: '/search', label: 'Search' },
  ];
  
  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center">
            <Link href="/" className="text-xl font-bold text-gray-900">
              TLDW
            </Link>
          </div>
          
          {/* Navigation */}
          <nav className="hidden md:flex md:space-x-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'inline-flex items-center px-1 pt-1 text-sm font-medium',
                  router.pathname === link.href
                    ? 'border-b-2 border-blue-500 text-gray-900'
                    : 'text-gray-500 hover:text-gray-900'
                )}
              >
                {link.label}
              </Link>
            ))}
          </nav>
          
          {/* User menu */}
          <div className="flex items-center space-x-4">
            {isAuthenticated ? (
              <>
                <span className="text-sm text-gray-700">
                  {user?.username || 'User'}
                </span>
                <button
                  onClick={handleLogout}
                  className="rounded-md bg-gray-100 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
                >
                  Logout
                </button>
              </>
            ) : (
              <Link
                href="/login"
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}