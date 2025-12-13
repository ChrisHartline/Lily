/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'aria-dark': '#0A0E1A',
        'aria-dark-blue': '#171F30',
        'aria-blue': '#3B82F6',
        'aria-gray': '#9CA3AF',
        'aria-light-gray': '#4B5563',
      }
    }
  },
  plugins: [],
}
