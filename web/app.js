"use strict";

// ============================================================
// 1. DOM state and message rendering
// ============================================================

const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const inputStatus = document.querySelector("#input-status");
const characterCount = document.querySelector("#character-count");
const promptChips = Array.from(document.querySelectorAll(".prompt-chip"));

const FRIENDLY_ERROR =
  "Sorry, AgentRec could not complete this request. Please try again.";

let isLoading = false;
let thinkingMessage = null;

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function appendInlineFormatting(parent, text) {
  const boldPattern = /\*\*(.+?)\*\*/g;
  let cursor = 0;
  let match;

  while ((match = boldPattern.exec(text)) !== null) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)));

    const strong = document.createElement("strong");
    strong.textContent = match[1];
    parent.append(strong);
    cursor = match.index + match[0].length;
  }

  parent.append(document.createTextNode(text.slice(cursor)));
}

function renderAssistantMarkdown(container, text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let paragraphLines = [];
  let list = null;
  let listType = null;

  function flushParagraph() {
    if (paragraphLines.length === 0) return;

    const paragraph = document.createElement("p");
    paragraphLines.forEach((line, index) => {
      if (index > 0) paragraph.append(document.createElement("br"));
      appendInlineFormatting(paragraph, line);
    });
    container.append(paragraph);
    paragraphLines = [];
  }

  function flushList() {
    if (list) container.append(list);
    list = null;
    listType = null;
  }

  lines.forEach((line) => {
    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const matchedListType = unorderedMatch ? "ul" : orderedMatch ? "ol" : null;
    const listItemText = unorderedMatch?.[1] ?? orderedMatch?.[1];

    if (headingMatch) {
      flushParagraph();
      flushList();

      const heading = document.createElement(`h${headingMatch[1].length}`);
      appendInlineFormatting(heading, headingMatch[2]);
      container.append(heading);
      return;
    }

    if (matchedListType) {
      flushParagraph();
      if (listType !== matchedListType) {
        flushList();
        list = document.createElement(matchedListType);
        listType = matchedListType;
      }

      const item = document.createElement("li");
      appendInlineFormatting(item, listItemText);
      list.append(item);
      return;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }

    flushList();
    paragraphLines.push(line);
  });

  flushParagraph();
  flushList();
  container.classList.add("has-markdown");
}

function addMessage(role, text, options = {}) {
  const article = document.createElement("article");
  const label = document.createElement("div");
  const bubble = document.createElement("div");

  article.className = `message message-${role}`;
  if (options.thinking) article.classList.add("message-thinking");
  if (options.error) article.classList.add("message-error");

  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "AgentRec";
  bubble.className = "message-bubble";
  if (role === "agent" && !options.thinking && !options.error) {
    // The formatter creates only allowlisted elements and inserts all LLM text
    // through text nodes/textContent, so raw HTML is never parsed or executed.
    renderAssistantMarkdown(bubble, text);
  } else {
    bubble.textContent = text;
  }

  article.append(label, bubble);
  messages.append(article);
  scrollToBottom();
  return article;
}


// ============================================================
// 2. Input and loading state
// ============================================================

function updateComposer() {
  characterCount.textContent = `${messageInput.value.length} / 4000`;
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
}

function setLoading(loading) {
  isLoading = loading;
  sendButton.disabled = loading;
  messageInput.disabled = loading;
  promptChips.forEach((chip) => {
    chip.disabled = loading;
  });
  messages.setAttribute("aria-busy", String(loading));

  if (loading) {
    inputStatus.textContent = "AgentRec is thinking…";
    thinkingMessage = addMessage("agent", "AgentRec is thinking…", {
      thinking: true,
    });
  } else {
    inputStatus.textContent = "Enter to send · Shift + Enter for a new line";
    if (thinkingMessage) {
      thinkingMessage.remove();
      thinkingMessage = null;
    }
  }
}


// ============================================================
// 3. Same-origin chat request
// ============================================================

async function sendMessage() {
  if (isLoading) return;

  const userMessage = messageInput.value.trim();
  if (!userMessage) {
    inputStatus.textContent = "Please enter a shopping request.";
    messageInput.focus();
    return;
  }

  addMessage("user", userMessage);
  messageInput.value = "";
  updateComposer();
  setLoading(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage }),
    });

    let payload;
    try {
      payload = await response.json();
    } catch (error) {
      console.error("AgentRec returned invalid JSON.", error);
      throw new Error("Invalid JSON response");
    }

    if (!response.ok) {
      console.error("AgentRec backend error:", payload?.detail ?? response.status);
      throw new Error(`HTTP ${response.status}`);
    }
    if (typeof payload.answer !== "string" || !payload.answer.trim()) {
      console.error("AgentRec response contained no answer.", payload);
      throw new Error("Missing answer");
    }

    // Reserved for a future debug panel; never displayed in the normal UI.
    const debugMetadata = {
      steps: payload.steps,
      toolCallCount: payload.tool_call_count,
      toolCalls: payload.tool_calls,
    };
    void debugMetadata;

    setLoading(false);
    addMessage("agent", payload.answer);
  } catch (error) {
    console.error("AgentRec request failed:", error);
    setLoading(false);
    addMessage("agent", FRIENDLY_ERROR, { error: true });
  } finally {
    messageInput.focus();
  }
}


// ============================================================
// 4. Keyboard, form, and example prompt interactions
// ============================================================

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendMessage();
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    void sendMessage();
  }
});

messageInput.addEventListener("input", updateComposer);

promptChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    messageInput.value = chip.textContent.trim();
    updateComposer();
    messageInput.focus();
  });
});

updateComposer();
