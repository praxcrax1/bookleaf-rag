import { redirect } from "next/navigation"
import { cookies } from "next/headers"
import { Header } from "@/components/header"
import { ChatWindow } from "@/components/chat-window"

export default async function ChatPage() {
  const token = cookies().get("token")?.value
  if (!token) {
    redirect("/login")
  }

  return (
    <main className="flex h-screen flex-col bg-white overflow-hidden">
      <Header />
      <section className="">
        <ChatWindow />
      </section>
    </main>
  )
}
