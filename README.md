# AI-Powered Resume Analyzer & Career Roadmap

A modern web application that combines AI-powered resume analysis with personalized career guidance. Upload your resume to get compatibility scores with job requirements and receive tailored career advice from an AI advisor.

## Features

### 🔍 Resume Analysis
- **Drag & Drop Upload**: Easy PDF resume upload with visual feedback
- **Job Matching**: Compare your resume against specific job requirements
- **Compatibility Scoring**: Get detailed percentage-based compatibility scores
- **Smart Recommendations**: Receive actionable feedback to improve your resume

### 🤖 AI Career Roadmap
- **Interactive Chat**: Conversational AI career advisor powered by Google's Gemini
- **Personalized Advice**: Career guidance based on your actual resume content
- **Skill Gap Analysis**: Identify missing skills and get learning recommendations
- **Career Path Planning**: Explore potential career transitions and growth opportunities

### 🎨 Modern UI/UX
- **Professional Design**: Clean, trustworthy interface with emerald green accent colors
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Dark Mode Support**: Toggle between light and dark themes
- **Accessible Components**: Built with accessibility best practices

## Tech Stack

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS v4** - Modern styling with design tokens
- **shadcn/ui** - High-quality component library
- **Radix UI** - Accessible primitive components
- **Lucide React** - Beautiful icon library

### Backend
- **Flask** - Python web framework
- **Google Gemini AI** - Advanced language model for career advice
- **spaCy** - Natural language processing
- **scikit-learn** - Machine learning for resume analysis
- **PyPDF2** - PDF text extraction

## Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+
- Google Gemini API key

### Frontend Setup

1. **Install dependencies**:
   \`\`\`bash
   npm install
   \`\`\`

2. **Start the development server**:
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

### Backend Setup

1. **Navigate to backend directory**:
   \`\`\`bash
   cd backend
   \`\`\`

2. **Install Python dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Download spaCy model**:
   \`\`\`bash
   python -m spacy download en_core_web_sm
   \`\`\`

4. **Set up environment variables**:
   \`\`\`bash
   cp .env.example .env
   # Add your Gemini API key to .env
   \`\`\`

5. **Start the Flask server**:
   \`\`\`bash
   python app.py
   \`\`\`

The backend will run on [http://localhost:5000](http://localhost:5000)

## API Endpoints

### Resume Management
- `POST /upload` - Upload PDF resume
- `POST /delete` - Delete uploaded resume

### Analysis & AI
- `POST /rate_resumes` - Analyze resume compatibility with job requirements
- `POST /career_roadmap` - Get AI-powered career advice

## Usage Guide

### 1. Resume Analysis
1. **Upload Resume**: Drag and drop your PDF resume or click to browse
2. **Add Job Requirements**: Paste the job description or requirements
3. **Analyze**: Click "Analyze Resume Match" to get your compatibility score
4. **Review Results**: See your score, recommendations, and areas for improvement

### 2. Career Roadmap
1. **Upload Resume**: Ensure you have a resume uploaded first
2. **Ask Questions**: Use the chat interface to ask about:
   - Skill development recommendations
   - Career advancement opportunities
   - Industry trends and insights
   - Certification suggestions
   - Career transition advice

## Project Structure

\`\`\`
├── app/                    # Next.js app directory
│   ├── globals.css        # Global styles with design tokens
│   ├── layout.tsx         # Root layout component
│   └── page.tsx           # Main application page
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   ├── resume-analyzer.tsx # Combined upload & analysis
│   └── career-roadmap.tsx  # AI chat interface
├── backend/              # Python Flask backend
│   ├── app.py           # Main Flask application
│   ├── requirements.txt # Python dependencies
│   └── .env.example     # Environment variables template
└── lib/                 # Utility functions
    └── utils.ts         # Helper functions
\`\`\`

## Design System

### Color Palette
- **Primary**: Emerald green (#059669) - Growth and success
- **Accent**: Bright emerald (#10b981) - Interactive elements
- **Neutrals**: Clean whites and grays for professional appearance
- **Background**: Pure white with subtle gray cards

### Typography
- **Headings**: Geist Sans - Clean, modern font for titles
- **Body**: Geist Sans - Excellent readability for content
- **Monospace**: Geist Mono - Code and technical content

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-username/resume-analyzer/issues) page
2. Create a new issue with detailed information
3. For urgent matters, contact the development team

---

Built with ❤️ using Next.js, Flask, and AI technology to help job seekers succeed in their careers.
