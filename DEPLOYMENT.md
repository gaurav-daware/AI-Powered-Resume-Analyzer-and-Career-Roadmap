# Deployment Guide

## Frontend Deployment (Vercel)

### Automatic Deployment
1. Push your code to GitHub
2. Connect your repository to Vercel
3. Vercel will automatically deploy on every push to main

### Manual Deployment
\`\`\`bash
npm run build
npx vercel --prod
\`\`\`

## Backend Deployment Options

### Option 1: Railway
1. Create account at [Railway](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables in Railway dashboard
4. Deploy automatically

### Option 2: Heroku
1. Install Heroku CLI
2. Create new Heroku app
3. Set environment variables
4. Deploy:
\`\`\`bash
cd backend
git init
heroku git:remote -a your-app-name
git add .
git commit -m "Deploy backend"
git push heroku main
\`\`\`

### Option 3: DigitalOcean App Platform
1. Create account at DigitalOcean
2. Use App Platform to deploy from GitHub
3. Configure environment variables
4. Set up automatic deployments

## Environment Variables

### Frontend (.env.local)
\`\`\`
NEXT_PUBLIC_API_URL=https://your-backend-url.com
\`\`\`

### Backend (.env)
\`\`\`
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_ENV=production
\`\`\`

## Production Considerations

### Security
- Use HTTPS for all communications
- Validate file uploads strictly
- Implement rate limiting
- Add CORS configuration for production domains

### Performance
- Enable gzip compression
- Use CDN for static assets
- Implement caching strategies
- Monitor API response times

### Monitoring
- Set up error tracking (Sentry)
- Monitor uptime and performance
- Log important events
- Set up alerts for failures
