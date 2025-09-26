"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Loader2, TrendingUp, FileText, AlertCircle } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface ResumeRatingProps {
  uploadedFile: string | null
}

interface RatingResult {
  filename: string
  score: number
  summary: string
}

interface RatingResponse {
  job_requirement: string
  resume_count: number
  results: RatingResult[]
}

export function ResumeRating({ uploadedFile }: ResumeRatingProps) {
  const [jobRequirement, setJobRequirement] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RatingResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { toast } = useToast()

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      toast({
        title: "No Resume",
        description: "Please upload a resume first",
        variant: "destructive",
      })
      return
    }

    if (!jobRequirement.trim()) {
      toast({
        title: "Missing Job Requirement",
        description: "Please enter a job description or requirements",
        variant: "destructive",
      })
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:5000/rate_resumes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          job_requirement: jobRequirement,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Analysis failed")
      }

      const data: RatingResponse = await response.json()
      setResult(data)

      toast({
        title: "Analysis Complete",
        description: `Your resume scored ${data.results[0]?.score}% compatibility`,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
      toast({
        title: "Analysis Failed",
        description: err instanceof Error ? err.message : "Analysis failed",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600"
    if (score >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 80) return "default"
    if (score >= 60) return "secondary"
    return "destructive"
  }

  return (
    <div className="space-y-6">
      {!uploadedFile && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Please upload a resume first to use the rating feature.</AlertDescription>
        </Alert>
      )}

      <div className="space-y-4">
        <div>
          <label htmlFor="job-requirement" className="block text-sm font-medium mb-2">
            Job Description / Requirements
          </label>
          <Textarea
            id="job-requirement"
            placeholder="Paste the job description or requirements here..."
            value={jobRequirement}
            onChange={(e) => setJobRequirement(e.target.value)}
            rows={8}
            className="w-full"
          />
        </div>

        <Button onClick={handleAnalyze} disabled={loading || !uploadedFile} className="w-full">
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Analyzing Resume...
            </>
          ) : (
            <>
              <TrendingUp className="h-4 w-4 mr-2" />
              Analyze Resume Compatibility
            </>
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {result.results.map((item, index) => (
              <div key={index} className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">{item.filename}</h3>
                  <Badge variant={getScoreBadgeVariant(item.score)}>{item.score}% Match</Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Compatibility Score</span>
                    <span className={`font-semibold ${getScoreColor(item.score)}`}>{item.score}%</span>
                  </div>
                  <Progress value={item.score} className="w-full" />
                </div>

                <div>
                  <h4 className="font-medium mb-2">Resume Summary</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">{item.summary}</p>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h4 className="font-medium mb-2 text-blue-800 dark:text-blue-200">Interpretation</h4>
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    {item.score >= 80 && "Excellent match! Your resume aligns very well with the job requirements."}
                    {item.score >= 60 &&
                      item.score < 80 &&
                      "Good match! Consider highlighting more relevant skills and experience."}
                    {item.score < 60 &&
                      "Room for improvement. Consider tailoring your resume to better match the job requirements."}
                  </p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}