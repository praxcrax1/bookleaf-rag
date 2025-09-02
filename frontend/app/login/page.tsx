import { redirect } from "next/navigation"
import dynamic from "next/dynamic"
import { cookies } from "next/headers"
import AuthFormClient from "./AuthFormClient"


export default async function LoginPage() {
  const token = cookies()?.get("token")?.value
  if (token) {
    redirect("/chat")
  }

  return (
    <main className="mx-auto flex min-h-[100svh] w-full max-w-6xl flex-col bg-white">
      <section className="flex flex-1 items-center justify-center p-4">
        <AuthFormClient />
      </section>
    </main>
  )
}
