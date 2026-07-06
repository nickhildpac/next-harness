import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cue — Conversation AI",
  description: "Conversation AI frontend"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
