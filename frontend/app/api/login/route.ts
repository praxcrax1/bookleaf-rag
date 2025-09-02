import { type NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const apiBase = process.env.API_BASE_URL
    if (!apiBase) {
      return NextResponse.json({ success: false, message: "API_BASE_URL not configured" }, { status: 500 })
    }

    const res = await fetch(`${apiBase}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    const data = await res.json().catch(() => ({}))
    if (!res.ok || data?.success === false || !data?.token) {
      return NextResponse.json(
        { success: false, message: data?.message ?? "Login failed" },
        { status: res.status || 401 },
      )
    }

    const response = NextResponse.json({ success: true, message: "Login successful" }, { status: 200 })
    response.cookies.set("token", data.token, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
      maxAge: 60 * 60 * 24 * 7, // 7 days
    })
    return response
  } catch (err: any) {
    return NextResponse.json({ success: false, message: err?.message ?? "Unexpected error" }, { status: 500 })
  }
}
