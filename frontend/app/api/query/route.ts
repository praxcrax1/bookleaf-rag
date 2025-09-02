import { type NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  const token = req.cookies.get("token")?.value
  if (!token) return NextResponse.json({ detail: "Unauthorized" }, { status: 401 })

  const apiBase = process.env.API_BASE_URL
  if (!apiBase) {
    return NextResponse.json({ detail: "API_BASE_URL not configured" }, { status: 500 })
  }

  try {
    const body = await req.json()
    const res = await fetch(`${apiBase}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    })
    const data = await res.json().catch(() => ({}))
    return NextResponse.json(data, { status: res.status })
  } catch (err: any) {
    return NextResponse.json({ detail: err?.message ?? "Unexpected error" }, { status: 500 })
  }
}
