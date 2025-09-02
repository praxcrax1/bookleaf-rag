import { type NextRequest, NextResponse } from "next/server"

export async function GET(req: NextRequest) {
  const token = req.cookies.get("token")?.value
  return NextResponse.json({ authenticated: Boolean(token) })
}
