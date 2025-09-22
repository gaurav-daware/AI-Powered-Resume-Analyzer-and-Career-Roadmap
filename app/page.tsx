"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResumeAnalyzer } from "@/components/resume-analyzer"
import { CareerRoadmap } from "@/components/career-roadmap"
import { Brain, FileText, TrendingUp } from "lucide-react"

export default function HomePage() {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="p-3 bg-primary/10 rounded-xl">
              <Brain className="h-10 w-10 text-primary" />
            </div>
            <h1 className="text-5xl font-bold text-foreground font-sans">AI Resume Analyzer</h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed font-sans">
            Upload your resume to get AI-powered insights, job compatibility scores, and personalized career roadmaps to
            accelerate your career growth
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto">
          <Tabs defaultValue="analyzer" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-8 h-14">
              <TabsTrigger value="analyzer" className="flex items-center gap-3 text-base font-medium">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  <TrendingUp className="h-5 w-5" />
                </div>
                Resume Analyzer
              </TabsTrigger>
              <TabsTrigger value="roadmap" className="flex items-center gap-3 text-base font-medium">
                <Brain className="h-5 w-5" />
                Career Roadmap
              </TabsTrigger>
            </TabsList>

            <TabsContent value="analyzer">
              <Card className="border-0 shadow-lg">
                <CardHeader className="pb-8">
                  <CardTitle className="text-2xl font-bold text-foreground">Resume Analysis & Job Matching</CardTitle>
                  <CardDescription className="text-lg text-muted-foreground">
                    Upload your resume and compare it against job requirements to get detailed compatibility insights
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResumeAnalyzer onUploadSuccess={setUploadedFile} />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="roadmap">
              <Card className="border-0 shadow-lg">
                <CardHeader className="pb-8">
                  <CardTitle className="text-2xl font-bold text-foreground">AI Career Roadmap</CardTitle>
                  <CardDescription className="text-lg text-muted-foreground">
                    Get personalized career advice, skill recommendations, and growth strategies based on your resume
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <CareerRoadmap uploadedFile={uploadedFile} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
