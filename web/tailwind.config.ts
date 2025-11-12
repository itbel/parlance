import type { Config } from "tailwindcss";
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          sky: "#38bdf8",
          emerald: "#34d399",
          amber: "#fbbf24"
        }
      },
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans]
      },
      boxShadow: {
        "glow-sky": "0 0 60px rgba(56, 189, 248, 0.25)"
      }
    }
  },
  plugins: []
};

export default config;
