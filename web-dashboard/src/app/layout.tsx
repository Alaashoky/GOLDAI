import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GOLDAI Dashboard",
  description: "AI-Powered XAUUSD Trading Bot",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-900 text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
