import { redirect } from "next/navigation"
import { cookies } from "next/headers"

export default async function RootPage() {
  const token = cookies().get("token")?.value
  redirect(token ? "/chat" : "/login")
}
