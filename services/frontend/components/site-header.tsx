"use client";

import { Activity, Aperture, Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/workspace", label: "Workspace" },
  { href: "/models", label: "Models" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-ink/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1480px] items-center justify-between px-5 lg:px-8">
        <Link href="/" className="group flex items-center gap-3" aria-label="DeepHorizon home">
          <span className="relative grid size-8 place-items-center rounded-full border border-ember/50 bg-ember/10">
            <Aperture className="size-4 text-ember transition-transform duration-500 group-hover:rotate-90" />
            <span className="absolute inset-1 rounded-full shadow-[0_0_18px_rgba(255,107,53,.4)]" />
          </span>
          <span className="font-mono text-sm font-semibold uppercase tracking-[0.18em] text-bone">
            Deep<span className="text-ember">Horizon</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-full px-4 py-2 text-sm text-zinc-400 transition hover:text-white",
                pathname.startsWith(link.href) && "bg-white/5 text-white",
              )}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1.5 md:flex">
          <Activity className="size-3.5 text-emerald-400" />
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-emerald-300">Systems nominal</span>
        </div>

        <button
          type="button"
          className="grid size-9 place-items-center rounded-lg border border-white/10 text-zinc-300 md:hidden"
          onClick={() => setIsOpen((open) => !open)}
          aria-label="Toggle navigation"
        >
          {isOpen ? <X className="size-4" /> : <Menu className="size-4" />}
        </button>
      </div>

      {isOpen && (
        <nav className="border-t border-white/10 bg-ink px-5 py-4 md:hidden">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setIsOpen(false)}
              className="block rounded-xl px-4 py-3 text-sm text-zinc-300 hover:bg-white/5"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
