"use client"

import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Upload,
  FileText,
  Trash2,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  Loader2,
  Target,
  BookOpen,
  Lightbulb,
  BarChart3,
} from "lucide-react"
import { useDropzone } from "react-dropzone"

interface ResumeAnalyzerProps {
  onUploadSuccess: (filename: string) => void
}

interface AnalysisResult {
  filename: string
  score: number
  summary: string
  ats_status: {
    level: string
    label: string
    color: string
  }
  keyword_analysis: {
    matching_keywords: string[]
    missing_keywords: string[]
    keyword_density: Record<string, number>
  }
  recommendations: string[]
  skill_gaps: {
    current_skills: string[]
    skill_gaps: Array<{
      skill: string
      importance: string
      resources: string[]
    }>
  }
}

export function ResumeAnalyzer({ onUploadSuccess }: ResumeAnalyzerProps) {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [jobRequirement, setJobRequirement] = useState("")
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const savedFile = localStorage.getItem("resume-analyzer-file")
    const savedJobRequirement = localStorage.getItem("resume-analyzer-job-requirement")
    const savedAnalysisResult = localStorage.getItem("resume-analyzer-analysis-result")

    if (savedFile) {
      setUploadedFile(savedFile)
      onUploadSuccess(savedFile)
    }
    if (savedJobRequirement) {
      setJobRequirement(savedJobRequirement)
    }
    if (savedAnalysisResult) {
      try {
        setAnalysisResult(JSON.parse(savedAnalysisResult))
      } catch (e) {
        console.error("Failed to parse saved analysis result:", e)
      }
    }
  }, [onUploadSuccess])

  useEffect(() => {
    if (uploadedFile) {
      localStorage.setItem("resume-analyzer-file", uploadedFile)
    } else {
      localStorage.removeItem("resume-analyzer-file")
    }
  }, [uploadedFile])

  useEffect(() => {
    if (jobRequirement) {
      localStorage.setItem("resume-analyzer-job-requirement", jobRequirement)
    } else {
      localStorage.removeItem("resume-analyzer-job-requirement")
    }
  }, [jobRequirement])

  useEffect(() => {
    if (analysisResult) {
      localStorage.setItem("resume-analyzer-analysis-result", JSON.stringify(analysisResult))
    } else {
      localStorage.removeItem("resume-analyzer-analysis-result")
    }
  }, [analysisResult])

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      console.log("[v0] Starting file upload:", file.name, file.type, file.size)

      setIsUploading(true)
      setError(null)

      const formData = new FormData()
      formData.append("file", file)

      try {
        console.log("[v0] Sending request to backend...")
        const response = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        })

        console.log("[v0] Response status:", response.status)

        if (response.ok) {
          const result = await response.json()
          console.log("[v0] Upload successful:", result)
          setUploadedFile(result.filename)
          onUploadSuccess(result.filename)
        } else {
          const errorData = await response.json()
          console.log("[v0] Upload error:", errorData)
          setError(errorData.error || "Upload failed")
        }
      } catch (err) {
        console.log("[v0] Network error:", err)
        setError(
          "Failed to connect to server. Please ensure:\n1. Backend server is running on http://localhost:5000\n2. CORS is properly configured\n3. File is a valid PDF",
        )
      } finally {
        setIsUploading(false)
      }
    },
    [onUploadSuccess],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB limit
    onDropRejected: (fileRejections) => {
      console.log("[v0] File rejected:", fileRejections)
      const rejection = fileRejections[0]
      if (rejection.errors.some((e) => e.code === "file-too-large")) {
        setError("File is too large. Please upload a PDF smaller than 10MB.")
      } else if (rejection.errors.some((e) => e.code === "file-invalid-type")) {
        setError("Invalid file type. Please upload a PDF file only.")
      } else {
        setError("File upload failed. Please try again.")
      }
    },
  })

  const handleDelete = async () => {
    try {
      await fetch("http://localhost:5000/delete", {
        method: "POST",
      })
      setUploadedFile(null)
      setAnalysisResult(null)
      setError(null)
      localStorage.removeItem("resume-analyzer-file")
      localStorage.removeItem("resume-analyzer-analysis-result")
    } catch (err) {
      setError("Failed to delete file")
    }
  }

  const handleAnalyze = async () => {
    if (!uploadedFile || !jobRequirement.trim()) return

    console.log("[v0] Starting analysis for:", uploadedFile)

    setIsAnalyzing(true)
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

      console.log("[v0] Analysis response status:", response.status)

      if (response.ok) {
        const result = await response.json()
        console.log("[v0] Analysis result:", result)
        setAnalysisResult(result.results[0])
      } else {
        const errorData = await response.json()
        console.log("[v0] Analysis error:", errorData)
        setError(errorData.error || "Analysis failed")
      }
    } catch (err) {
      console.log("[v0] Analysis network error:", err)
      setError("Failed to analyze resume. Make sure the backend is running.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-emerald-600"
    if (score >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const getScoreBadge = (score: number) => {
    if (score >= 80) return { label: "Excellent Match", variant: "default" as const }
    if (score >= 60) return { label: "Good Match", variant: "secondary" as const }
    return { label: "Needs Improvement", variant: "destructive" as const }
  }

  const getStatusColorClasses = (color: string) => {
    switch (color) {
      case "green":
        return "bg-emerald-100 text-emerald-800 border-emerald-200"
      case "yellow":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "red":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="grid md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5 text-primary" />
              Upload Resume
            </CardTitle>
            <CardDescription>Upload your PDF resume to get started with the analysis</CardDescription>
          </CardHeader>
          <CardContent>
            {!uploadedFile ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50 hover:bg-muted/50"
                }`}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-4">
                  <div className="p-4 bg-primary/10 rounded-full">
                    <FileText className="h-8 w-8 text-primary" />
                  </div>
                  {isUploading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Uploading...</span>
                    </div>
                  ) : (
                    <>
                      <div>
                        <p className="text-lg font-medium">
                          {isDragActive ? "Drop your resume here" : "Drag & drop your resume"}
                        </p>
                        <p className="text-sm text-muted-foreground mt-1">or click to browse (PDF only)</p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-emerald-600" />
                  <div>
                    <p className="font-medium">{uploadedFile}</p>
                    <p className="text-sm text-muted-foreground">Resume uploaded successfully</p>
                  </div>
                </div>
                <Button variant="outline" size="sm" onClick={handleDelete}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Job Requirements
            </CardTitle>
            <CardDescription>Paste the job description or requirements to compare against your resume</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="Paste the job description, requirements, or skills needed for the position you're applying to..."
              value={jobRequirement}
              onChange={(e) => setJobRequirement(e.target.value)}
              className="min-h-[200px] resize-none"
            />
            <Button
              onClick={handleAnalyze}
              disabled={!uploadedFile || !jobRequirement.trim() || isAnalyzing}
              className="w-full mt-4"
              size="lg"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing Resume...
                </>
              ) : (
                <>
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Analyze Resume Match
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Enhanced Analysis Results */}
      {analysisResult && (
        <div className="space-y-6">
          {/* ATS Compatibility Score & Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                ATS Compatibility Score & Summary
              </CardTitle>
              <CardDescription>How well your resume matches the job requirements and ATS systems</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold">ATS Score</h3>
                  <p className="text-muted-foreground">Applicant Tracking System compatibility</p>
                </div>
                <div className="text-right">
                  <div className={`text-4xl font-bold ${getScoreColor(analysisResult.score)}`}>
                    {analysisResult.score}%
                  </div>
                  <div
                    className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border mt-2 ${getStatusColorClasses(analysisResult.ats_status.color)}`}
                  >
                    {analysisResult.ats_status.label}
                  </div>
                </div>
              </div>

              <div>
                <Progress value={analysisResult.score} className="h-3" />
              </div>

              <div>
                <h4 className="font-semibold mb-3">Resume Summary</h4>
                <p className="text-muted-foreground leading-relaxed">{analysisResult.summary}</p>
              </div>
            </CardContent>
          </Card>

          {/* Keyword Analysis */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Keyword Analysis
              </CardTitle>
              <CardDescription>Detailed breakdown of keyword matches and gaps</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Top Matching Keywords */}
              <div>
                <h4 className="font-semibold mb-3 text-emerald-700">Top Matching Keywords</h4>
                <div className="flex flex-wrap gap-2">
                  {analysisResult.keyword_analysis.matching_keywords.slice(0, 12).map((keyword, index) => (
                    <Badge key={index} variant="secondary" className="bg-emerald-100 text-emerald-800">
                      {keyword}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Missing Keywords */}
              {analysisResult.keyword_analysis.missing_keywords.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-3 text-red-700">Missing Keywords</h4>
                  <div className="flex flex-wrap gap-2">
                    {analysisResult.keyword_analysis.missing_keywords.slice(0, 8).map((keyword, index) => (
                      <Badge key={index} variant="outline" className="border-red-200 text-red-700">
                        {keyword}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Keyword Density */}
              {Object.keys(analysisResult.keyword_analysis.keyword_density).length > 0 && (
                <div>
                  <h4 className="font-semibold mb-3">Keyword Density</h4>
                  <div className="space-y-2">
                    {Object.entries(analysisResult.keyword_analysis.keyword_density)
                      .slice(0, 6)
                      .map(([keyword, count]) => (
                        <div key={keyword} className="flex items-center justify-between">
                          <span className="text-sm font-medium">{keyword}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-primary h-2 rounded-full"
                                style={{
                                  width: `${Math.min((count / Math.max(...Object.values(analysisResult.keyword_analysis.keyword_density))) * 100, 100)}%`,
                                }}
                              ></div>
                            </div>
                            <span className="text-sm text-muted-foreground w-8">{count}</span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recommendations & Improvements */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-primary" />
                Recommendations & Improvements
              </CardTitle>
              <CardDescription>Prioritized actions to improve your resume</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysisResult.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                    <div className="flex-shrink-0 w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-medium">
                      {index + 1}
                    </div>
                    <p className="text-sm leading-relaxed">{recommendation}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Career Roadmap & Skill Gap Analysis */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-primary" />
                Career Roadmap & Skill Gap Analysis
              </CardTitle>
              <CardDescription>Long-term career planning and skill development</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Current Skills */}
              <div>
                <h4 className="font-semibold mb-3">Identified Skills</h4>
                <div className="flex flex-wrap gap-2">
                  {analysisResult.skill_gaps.current_skills.map((skill, index) => (
                    <Badge key={index} variant="default" className="bg-blue-100 text-blue-800">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Skill Gaps */}
              {analysisResult.skill_gaps.skill_gaps.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-3">Skill Gaps & Learning Resources</h4>
                  <div className="space-y-4">
                    {analysisResult.skill_gaps.skill_gaps.map((gap, index) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-medium">{gap.skill}</h5>
                          <Badge
                            variant={
                              gap.importance === "high"
                                ? "destructive"
                                : gap.importance === "medium"
                                  ? "secondary"
                                  : "outline"
                            }
                            className="text-xs"
                          >
                            {gap.importance} priority
                          </Badge>
                        </div>
                        <div className="space-y-1">
                          <p className="text-sm text-muted-foreground mb-2">Recommended learning resources:</p>
                          {gap.resources.map((resource, resourceIndex) => (
                            <div key={resourceIndex} className="flex items-center gap-2 text-sm">
                              <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                              <span>{resource}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
