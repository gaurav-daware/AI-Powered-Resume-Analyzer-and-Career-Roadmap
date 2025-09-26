"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User, AlertCircle, Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface CareerRoadmapProps {
  uploadedFile: string | null
}

interface Message {
  id: string
  type: "user" | "bot"
  content: string
  timestamp: Date
}

export function CareerRoadmap({ uploadedFile }: CareerRoadmapProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  useEffect(() => {
    // Load messages from localStorage on component mount
    const savedMessages = localStorage.getItem("career-roadmap-messages")
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }))
        setMessages(parsedMessages)
      } catch (error) {
        console.error("Failed to load saved messages:", error)
      }
    }
  }, [])

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("career-roadmap-messages", JSON.stringify(messages))
    }
  }, [messages])

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  // utility to clean unwanted markdown symbols
const cleanMessage = (text: string) => {
  return text
    .replace(/^##\s?/gm, "")      // remove headings ##
    .replace(/\*\*/g, "")         // remove bold markers
    .replace(/\*/g, "")           // remove bullet markers *
    .trim();
};


  const handleSendMessage = async () => {
    if (!uploadedFile) {
      toast({
        title: "No Resume",
        description: "Please upload a resume first",
        variant: "destructive",
      })
      return
    }

    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:5000/career_roadmap", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userMessage.content,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to get career advice")
      }

      const data = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: data.response,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get career advice")
      toast({
        title: "Request Failed",
        description: err instanceof Error ? err.message : "Failed to get career advice",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChatHistory = () => {
    setMessages([])
    localStorage.removeItem("career-roadmap-messages")
    toast({
      title: "Chat Cleared",
      description: "Chat history has been cleared",
    })
  }

  const suggestedQuestions = [
    "What skills should I develop to advance my career?",
    "What are the next steps in my career path?",
    "How can I transition to a different role?",
    "What certifications would benefit my career?",
    "What are the current market trends in my field?",
  ]

  return (
    <div className="space-y-4">
      {!uploadedFile && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Please upload a resume first to get personalized career advice.</AlertDescription>
        </Alert>
      )}

      {messages.length === 0 && uploadedFile && (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center mb-6">
              <Bot className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">AI Career Advisor</h3>
              <p className="text-gray-600 dark:text-gray-300">
                Ask me anything about your career path, skill development, or job market trends
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Try asking:</p>
              <div className="grid gap-2">
                {suggestedQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    className="justify-start text-left h-auto py-2 px-3 bg-transparent"
                    onClick={() => setInput(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {messages.length > 0 && (
        <Card>
          <CardContent className="p-0">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="font-semibold">Career Chat</h3>
              <Button variant="outline" size="sm" onClick={clearChatHistory} className="text-xs bg-transparent">
                Clear Chat
              </Button>
            </div>
            <ScrollArea className="h-96 p-4" ref={scrollAreaRef}>
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.type === "bot" && (
                      <div className="flex-shrink-0">
                        <Bot className="h-8 w-8 text-blue-600 bg-blue-100 dark:bg-blue-900 rounded-full p-1.5" />
                      </div>
                    )}

                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        message.type === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                      }`}
                    >
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">{cleanMessage(message.content)}</div>
                      <div className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</div>
                    </div>

                    {message.type === "user" && (
                      <div className="flex-shrink-0">
                        <User className="h-8 w-8 text-gray-600 bg-gray-200 dark:bg-gray-700 rounded-full p-1.5" />
                      </div>
                    )}
                  </div>
                ))}

                {loading && (
                  <div className="flex gap-3 justify-start">
                    <div className="flex-shrink-0">
                      <Bot className="h-8 w-8 text-blue-600 bg-blue-100 dark:bg-blue-900 rounded-full p-1.5" />
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-2">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Input
          placeholder="Ask about your career path, skills, or job market trends..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading || !uploadedFile}
          className="flex-1"
        />
        <Button onClick={handleSendMessage} disabled={loading || !input.trim() || !uploadedFile}>
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}