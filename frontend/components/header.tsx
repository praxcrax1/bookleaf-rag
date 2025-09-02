"use client"

import { useRouter, usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useState } from "react"

export function Header({ className }: { className?: string }) {
  const router = useRouter()
  const pathname = usePathname()
  const [loading, setLoading] = useState(false)

  async function onLogout() {
    setLoading(true)
    try {
      await fetch("/api/logout", { method: "POST" })
      router.replace("/login")
    } finally {
      setLoading(false)
    }
  }

  const showLogout = pathname?.startsWith("/chat")

  return (
    <header className={cn("w-full border-b bg-white", className)}>
      <div className="mx-auto flex max-w-4xl items-center justify-between p-4">
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-sm bg-emerald-600" aria-hidden="true" />
          <span className="font-semibold">Bookleaf Support</span>
        </div>
        {showLogout && (
          <Button variant="outline" onClick={onLogout} disabled={loading} className="border-slate-300 bg-transparent">
            {loading ? "Logging out..." : "Logout"}
          </Button>
        )}
      </div>
    </header>
  )
}
