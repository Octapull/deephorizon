import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#090a0c",
        panel: "#111318",
        ember: "#ff6b35",
        solar: "#ffad33",
        bone: "#f3efe6",
      },
      boxShadow: {
        glow: "0 0 48px rgba(255, 107, 53, 0.18)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
