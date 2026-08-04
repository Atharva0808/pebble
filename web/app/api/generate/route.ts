import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { prompt, max_tokens, temperature } = await req.json();

    if (!prompt) {
      return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
    }

    // Try calling the local PyTorch server running checkpoint_step_1500.pt
    try {
      const serverRes = await fetch("http://localhost:8000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens, temperature }),
        cache: "no-store",
      });

      if (serverRes.ok) {
        const data = await serverRes.json();
        return NextResponse.json(data);
      }
    } catch {
      // Server offline fallback
    }

    return NextResponse.json(
      { error: "PyTorch server offline", fallback: true },
      { status: 503 }
    );
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
