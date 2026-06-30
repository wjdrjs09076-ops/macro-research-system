import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Macro Research Portal",
  description: "Ontology-Gated Framework (ML + event_vol in shadow eval) — Crisis Signal Visualization",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-8">
            <span className="text-sm font-semibold text-emerald-400 tracking-wide">
              MACRO RESEARCH
            </span>
            <Link href="/" className="text-sm text-gray-400 hover:text-gray-100 transition-colors">
              Architecture
            </Link>
            <Link href="/ontology" className="text-sm text-gray-400 hover:text-gray-100 transition-colors">
              Ontology Trace
            </Link>
            <Link href="/gate" className="text-sm text-gray-400 hover:text-gray-100 transition-colors">
              Gate Timeline
            </Link>
            <Link href="/evt" className="text-sm text-gray-400 hover:text-gray-100 transition-colors">
              EVT Bridge
            </Link>
            <Link href="/analyze" className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors font-semibold">
              이벤트 분석기 ✦
            </Link>
            <Link href="/monitor" className="text-sm text-gray-400 hover:text-gray-100 transition-colors">
              Vol Monitor
            </Link>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
