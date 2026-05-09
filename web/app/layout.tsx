import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pebble",
  description:
    "A 120M parameter general-purpose language model built from scratch on the Mamba-2 Selective State Space architecture. Linear-time inference. Zero-cost training. Open source.",
  keywords: [
    "Pebble",
    "Mamba",
    "State Space Model",
    "SLM",
    "Language Model",
    "Machine Learning",
    "PyTorch",
    "SSM",
  ],
  authors: [{ name: "Atharva" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;450;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
