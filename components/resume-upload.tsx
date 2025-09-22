"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileText, CheckCircle, XCircle, Trash2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface ResumeUploadProps {
  onUploadSuccess: (filename: string) => void
}

export function ResumeUpload({ onUploadSuccess }: ResumeUploadProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { toast } = useToast()

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      if (file.type !== "application/pdf") {
        setError("Please upload a PDF file only")
        return
      }

      setError(null)
      setUploading(true)
      setUploadProgress(0)

      const formData = new FormData()
      formData.append("file", file)

      try {
        // Simulate progress
        const progressInterval = setInterval(() => {
          setUploadProgress((prev) => Math.min(prev + 10, 90))
        }, 100)

        const response = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        })

        clearInterval(progressInterval)
        setUploadProgress(100)

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || "Upload failed")
        }

        const data = await response.json()
        setUploadedFile(data.filename)
        onUploadSuccess(data.filename)

        toast({
          title: "Success!",
          description: "Resume uploaded and processed successfully",
        })
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed")
        toast({
          title: "Upload Failed",
          description: err instanceof Error ? err.message : "Upload failed",
          variant: "destructive",
        })
      } finally {
        setUploading(false)
        setTimeout(() => setUploadProgress(0), 1000)
      }
    },
    [onUploadSuccess, toast],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    multiple: false,
    disabled: uploading,
  })

  const handleDelete = async () => {
    try {
      const response = await fetch("http://localhost:5000/delete", {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error("Delete failed")
      }

      setUploadedFile(null)
      onUploadSuccess("")
      toast({
        title: "Deleted",
        description: "Resume deleted successfully",
      })
    } catch (err) {
      toast({
        title: "Delete Failed",
        description: err instanceof Error ? err.message : "Delete failed",
        variant: "destructive",
      })
    }
  }

  if (uploadedFile) {
    return (
      <Card className="border-green-200 bg-green-50 dark:bg-green-900/20">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-8 w-8 text-green-600" />
              <div>
                <h3 className="font-semibold text-green-800 dark:text-green-200">Resume Uploaded Successfully</h3>
                <p className="text-sm text-green-600 dark:text-green-300">{uploadedFile}</p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDelete}
              className="text-red-600 hover:text-red-700 bg-transparent"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card
        {...getRootProps()}
        className={`border-2 border-dashed cursor-pointer transition-colors ${
          isDragActive ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-gray-300 hover:border-gray-400"
        } ${uploading ? "pointer-events-none opacity-50" : ""}`}
      >
        <CardContent className="pt-6">
          <input {...getInputProps()} />
          <div className="text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <div className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              {isDragActive ? "Drop your resume here" : "Upload your resume"}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              Drag and drop your PDF resume, or click to browse
            </p>
            <Button disabled={uploading}>
              <FileText className="h-4 w-4 mr-2" />
              Choose File
            </Button>
          </div>
        </CardContent>
      </Card>

      {uploading && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Uploading and processing...</span>
            <span>{uploadProgress}%</span>
          </div>
          <Progress value={uploadProgress} className="w-full" />
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}
