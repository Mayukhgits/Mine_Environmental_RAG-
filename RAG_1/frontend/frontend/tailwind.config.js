/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f7ff',
          500: '#0284c7',
          900: '#0c4a6e',
        }
      }
    },
  },
  plugins: [],
}