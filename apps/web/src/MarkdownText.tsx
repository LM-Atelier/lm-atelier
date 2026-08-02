import { isValidElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CopyTextButton } from "./CopyTextButton";

function nodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return nodeText(node.props.children);
  return "";
}

export function MarkdownText({ text }: { text: string }) {
  return (
    <div className="message-text markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, href, ...props }) => {
            const external = /^https?:\/\//i.test(href ?? "");
            return (
              <a
                {...props}
                href={href}
                {...(external
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
              >
                {children}
              </a>
            );
          },
          img: ({ alt }) => <span className="markdown-image-reference">[Image: {alt || "link"}]</span>,
          pre: ({ children }) => {
            const code = nodeText(children).replace(/\n$/, "");
            return (
              <div className="markdown-code-block">
                {code && <CopyTextButton text={code} label="Copy code block" className="block-copy" />}
                <pre>{children}</pre>
              </div>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
