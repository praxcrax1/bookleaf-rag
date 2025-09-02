"use client"

import type React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

type ToolStep = {
  tool: string
  input?: string
  output?: string
}

type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  tools?: ToolStep[] // optional tools, shown only for assistant messages when present
}

export function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  async function sendMessage() {
    const question = input.trim()
    if (!question) return
    setInput("")
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: question }
    setMessages((m) => [...m, userMsg])
    setSending(true)

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, verbose: false }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const errorText = data?.detail || data?.message || "Error occurred. Please login again."
        setMessages((m) => [...m, { id: crypto.randomUUID(), role: "assistant", content: errorText }])
        return
      }
      const answer = data?.answer ?? "No answer returned."
      const tools: ToolStep[] | undefined = Array.isArray(data?.reasoning_steps)
        ? data.reasoning_steps.map((s: any) => ({
            tool: String(s?.tool ?? s?.name ?? "tool"),
            input:
              typeof s?.input === "string" ? s.input : s?.input != null ? JSON.stringify(s.input, null, 2) : undefined,
            output:
              typeof s?.output === "string"
                ? s.output
                : s?.output != null
                  ? JSON.stringify(s.output, null, 2)
                  : undefined,
          }))
        : undefined

      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: answer,
          tools,
        },
      ])
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), role: "assistant", content: e?.message ?? "Unexpected error" },
      ])
    } finally {
      setSending(false)
      queueMicrotask(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" })
      })
    }
  }

  async function deleteHistory() {
    const ok = confirm("Delete your chat history on the server?")
    if (!ok) return
    try {
      const res = await fetch("/api/query/delete", { method: "DELETE" })
      const data = await res.json().catch(() => ({}))
      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data?.message ?? "Chat history cleared.",
        },
      ])
    } catch {
      setMessages([{ id: crypto.randomUUID(), role: "assistant", content: "Failed to delete history." }])
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex h-[calc(100vh-64px)] flex-col">
      <div
        ref={listRef}
        className="mx-auto w-full max-w-3xl flex-1 overflow-y-auto p-4"
        aria-live="polite"
        aria-busy={sending}
      >
        {messages.length === 0 ? (
          <Card className="bg-white">
            <CardContent className="p-6">
              <h2 className="mb-1 text-lg font-semibold">Welcome to Bookleaf Support</h2>
              <p className="text-sm text-slate-600">
                Ask about publishing plans, timelines, or your order. Your messages are processed by our assistant
                trained on Bookleaf Publishing resources.
              </p>
            </CardContent>
          </Card>
        ) : (
          <ul className="grid gap-3">
            {messages.map((m) => (
              <li
                key={m.id}
                className={cn("rounded-md p-3", m.role === "user" ? "bg-emerald-50 text-slate-900" : "bg-white border")}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="whitespace-pre-wrap text-pretty leading-relaxed">{children}</p>,
                    a: ({ node, ...props }) => (
                      <a
                        {...props}
                        className="text-emerald-700 underline hover:text-emerald-800"
                        target="_blank"
                        rel="noopener noreferrer"
                      />
                    ),
                    ul: ({ children }) => <ul className="my-2 list-disc pl-5">{children}</ul>,
                    ol: ({ children }) => <ol className="my-2 list-decimal pl-5">{children}</ol>,
                    li: ({ children }) => <li className="my-1">{children}</li>,
                    blockquote: ({ children }) => (
                      <blockquote className="my-3 border-l-4 border-emerald-200 bg-emerald-50/40 pl-3 italic text-slate-700">
                        {children}
                      </blockquote>
                    ),
                    hr: () => <hr className="my-4 border-slate-200" />,
                    h1: ({ children }) => <h1 className="mb-2 text-xl font-semibold">{children}</h1>,
                    h2: ({ children }) => <h2 className="mb-2 text-lg font-semibold">{children}</h2>,
                    h3: ({ children }) => <h3 className="mb-2 text-base font-semibold">{children}</h3>,
                    code({ className, children, ...props }) {

                      return (
                        <pre className="my-3 overflow-x-auto rounded-md bg-slate-900 p-3 text-slate-100">
                          <code className="font-mono text-sm" {...props}>
                            {children}
                          </code>
                        </pre>
                      )
                    },
                  }}
                >
                  {m.content}
                </ReactMarkdown>

                {m.role === "assistant" && m.tools && m.tools.length > 0 ? (
                  <Accordion type="single" collapsible className="mt-3">
                    <AccordionItem value={`tools-${m.id}`}>
                      <AccordionTrigger className="text-sm">{"Tools used (" + m.tools.length + ")"}</AccordionTrigger>
                      <AccordionContent>
                        <ul className="space-y-3">
                          {m.tools.map((t, idx) => (
                            <li key={idx} className="rounded-md border bg-slate-50 p-3">
                              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                                Tool
                              </div>
                              <div className="mb-2 font-medium text-slate-900">{t.tool}</div>

                              {t.input ? (
                                <div className="mb-2">
                                  <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                                    Input
                                  </div>
                                  <pre className="max-h-48 overflow-auto rounded bg-slate-900 p-3 text-xs text-slate-100 whitespace-pre-wrap">
                                    {t.input}
                                  </pre>
                                </div>
                              ) : null}

                              {t.output ? (
                                <div>
                                  <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                                    Output
                                  </div>
                                  <pre className="max-h-48 overflow-auto rounded bg-slate-900 p-3 text-xs text-slate-100 whitespace-pre-wrap">
                                    {t.output}
                                  </pre>
                                </div>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t bg-white">
        <div className="mx-auto flex w-full max-w-3xl items-center gap-2 p-4">
          <Input
            placeholder="Type your question about Bookleaf Publishing..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Message input"
          />
          <Button
            onClick={sendMessage}
            disabled={sending || input.trim().length === 0}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            {sending ? "Sending..." : "Send"}
          </Button>
          <Button variant="outline" onClick={deleteHistory} className="border-slate-300 text-red-600 bg-transparent">
            Delete chat
          </Button>
        </div>
      </div>
    </div>
  )
}
