"use strict";

// ============================================================
// 1. DOM state and safe message rendering
// ============================================================

const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const inputStatus = document.querySelector("#input-status");
const characterCount = document.querySelector("#character-count");
const statusBadge = document.querySelector("#agent-status-badge");
const planContent = document.querySelector("#shopping-plan-content");
const promptChips = Array.from(document.querySelectorAll(".prompt-chip"));

const FRIENDLY_ERROR =
  "Sorry, AgentRec could not complete this request. Please try again.";
const FRIENDLY_READY =
  "Your shopping plan is ready. Review the selected products and budget on the right.";
const FRIENDLY_CLARIFICATION =
  "Please provide the missing information so I can build your shopping plan.";
const FRIENDLY_CONFLICT =
  "The current shopping requirements need adjustment. Review the plan details on the right.";

const STATUS_LABELS = {
  ready: "Ready",
  clarification_required: "Needs input",
  conflict: "Needs adjustment",
  error: "Error",
};

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
// 2. Web-safe Shopping Plan projection
// ============================================================

function createTextElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function formatMoney(value, currency = "USD") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const amount = value.toFixed(2);
  return currency === "USD" ? `$${amount}` : `${currency} ${amount}`;
}

function humanizeStatus(value) {
  if (typeof value !== "string" || !value) return "Unknown";
  return value.replaceAll("_", " ");
}

function updateStatusBadge(status) {
  const label = STATUS_LABELS[status];
  if (!label) {
    statusBadge.hidden = true;
    statusBadge.removeAttribute("data-status");
    statusBadge.textContent = "";
    return;
  }
  statusBadge.hidden = false;
  statusBadge.dataset.status = status;
  statusBadge.textContent = label;
}

function renderPlanEmpty(title, detail) {
  const empty = document.createElement("div");
  const icon = createTextElement("div", "empty-state-icon", "＋");
  const heading = createTextElement("p", "", title);
  const description = createTextElement("span", "", detail);
  icon.setAttribute("aria-hidden", "true");
  empty.className = "plan-empty-state";
  empty.append(icon, heading, description);
  planContent.replaceChildren(empty);
}

function createMetric(label, value) {
  const card = document.createElement("div");
  card.className = "metric-card";
  card.append(
    createTextElement("span", "metric-label", label),
    createTextElement("span", "metric-value", value),
  );
  return card;
}

function appendConflictList(panel, label, values) {
  if (!Array.isArray(values) || values.length === 0) return;
  panel.append(createTextElement("p", "", label));
  const list = document.createElement("ul");
  list.className = "conflict-list";
  values.forEach((value) => {
    list.append(createTextElement("li", "", String(value)));
  });
  panel.append(list);
}

function createConflictPanel(conflict) {
  const panel = document.createElement("section");
  panel.className = "conflict-panel plan-section";
  panel.append(
    createTextElement("h3", "", "Plan needs adjustment"),
    createTextElement("p", "", `Affected category: ${String(conflict.category ?? "")}`),
  );
  appendConflictList(panel, "Required features", conflict.required_features);
  appendConflictList(panel, "Constraints not satisfied", conflict.failed_constraints);
  appendConflictList(panel, "Evidence still unknown", conflict.unknown_constraints);
  appendConflictList(panel, "Evidence contradicted", conflict.contradicted_constraints);
  if (
    typeof conflict.replan_attempts_performed === "number" &&
    Number.isFinite(conflict.replan_attempts_performed)
  ) {
    panel.append(
      createTextElement(
        "p",
        "",
        `Bounded re-plan attempts: ${conflict.replan_attempts_performed}`,
      ),
    );
  }
  return panel;
}

function createProductCard(product, requirement, currency) {
  const card = document.createElement("article");
  const header = document.createElement("header");
  const status = requirement?.status ?? "unknown";
  const facts = document.createElement("dl");
  card.className = "product-card";
  header.className = "product-card-header";
  facts.className = "product-facts";

  const category = createTextElement(
    "span",
    "product-category",
    String(product.category ?? ""),
  );
  const statusBadgeElement = createTextElement(
    "span",
    "requirement-status",
    humanizeStatus(status),
  );
  statusBadgeElement.dataset.status = status;
  header.append(category, statusBadgeElement);

  const factValues = [
    ["Price", formatMoney(product.price, currency)],
    ["Quantity", String(product.quantity ?? "")],
    ["Subtotal", formatMoney(product.subtotal, currency)],
  ];
  factValues.forEach(([label, value]) => {
    const fact = document.createElement("div");
    fact.className = "product-fact";
    fact.append(
      createTextElement("dt", "", label),
      createTextElement("dd", "", value),
    );
    facts.append(fact);
  });

  card.append(
    header,
    createTextElement("h3", "product-title", String(product.title ?? "")),
    facts,
    createTextElement(
      "p",
      "product-id",
      `Product ID: ${String(product.parent_asin ?? "")}`,
    ),
  );

  const verified = Array.isArray(product.verified_requirements)
    ? product.verified_requirements.filter(
        (item) => item && item.status === "supported",
      )
    : [];
  if (verified.length > 0) {
    const badges = document.createElement("div");
    badges.className = "verified-list";
    verified.forEach((item) => {
      badges.append(
        createTextElement(
          "span",
          "verified-badge",
          `✓ ${String(item.constraint ?? "")} verified`,
        ),
      );
    });
    card.append(badges);
  }
  return card;
}

function renderShoppingPlan(payload) {
  const plan = payload?.plan;
  const requirements = Array.isArray(payload?.requirements)
    ? payload.requirements
    : [];
  const products = Array.isArray(payload?.products) ? payload.products : [];
  const conflict = payload?.conflict;

  if (!plan && !conflict) {
    if (payload?.status === "clarification_required") {
      renderPlanEmpty(
        "More information needed",
        "Answer AgentRec’s question to create a grounded plan.",
      );
    } else if (payload?.status === "error") {
      renderPlanEmpty(
        "No plan available",
        "AgentRec could not safely produce a shopping plan.",
      );
    } else {
      renderPlanEmpty(
        "Your shopping plan will appear here.",
        "Start with a goal, budget, and the products you need.",
      );
    }
    return;
  }

  const fragment = document.createDocumentFragment();
  if (conflict && typeof conflict === "object") {
    fragment.append(createConflictPanel(conflict));
  }

  if (plan && typeof plan === "object") {
    const summary = document.createElement("section");
    const heading = document.createElement("div");
    const metrics = document.createElement("div");
    const currency = typeof plan.currency === "string" ? plan.currency : "USD";
    summary.className = "plan-section";
    heading.className = "plan-section-heading";
    metrics.className = "plan-metrics";
    heading.append(
      createTextElement("h3", "", "Plan Summary"),
      createTextElement("span", "", humanizeStatus(plan.status)),
    );
    metrics.append(
      createMetric("Total budget", formatMoney(plan.total_budget, currency)),
      createMetric("Spent", formatMoney(plan.total_spent, currency)),
      createMetric("Remaining", formatMoney(plan.remaining_budget, currency)),
    );

    const track = document.createElement("div");
    const progress = document.createElement("div");
    const validTotal =
      typeof plan.total_budget === "number" &&
      Number.isFinite(plan.total_budget) &&
      plan.total_budget > 0;
    const validSpent =
      typeof plan.total_spent === "number" && Number.isFinite(plan.total_spent);
    const ratio = validTotal && validSpent ? plan.total_spent / plan.total_budget : 0;
    const progressPercent = Math.min(100, Math.max(0, ratio * 100));
    track.className = "budget-track";
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(Math.round(progressPercent)));
    progress.className = "budget-progress";
    if (ratio > 1) progress.classList.add("is-over-budget");
    progress.style.width = `${progressPercent}%`;
    track.append(progress);

    const satisfiedCount = requirements.filter(
      (requirement) => requirement?.status === "satisfied",
    ).length;
    const requirementProgress = createTextElement(
      "p",
      "requirement-progress",
      `${satisfiedCount} of ${requirements.length} requirements satisfied`,
    );
    summary.append(heading, metrics, track, requirementProgress);
    fragment.append(summary);

    if (products.length > 0) {
      const productsSection = document.createElement("section");
      const productsHeading = document.createElement("div");
      const productList = document.createElement("div");
      const requirementsById = new Map(
        requirements.map((requirement) => [requirement.requirement_id, requirement]),
      );
      productsSection.className = "plan-section";
      productsHeading.className = "plan-section-heading";
      productList.className = "product-list";
      productsHeading.append(
        createTextElement("h3", "", "Selected Products"),
        createTextElement("span", "", `${products.length} items`),
      );
      products.forEach((product) => {
        productList.append(
          createProductCard(
            product,
            requirementsById.get(product.requirement_id),
            currency,
          ),
        );
      });
      productsSection.append(productsHeading, productList);
      fragment.append(productsSection);
    }
  }

  planContent.replaceChildren(fragment);
}


// ============================================================
// 3. Input and loading state
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
// 4. Same-origin chat request
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
  updateStatusBadge(null);
  setLoading(true);

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: "web-demo",
        query: userMessage,
      }),
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

    let assistantText;
    switch (payload.status) {
      case "ready":
        assistantText =
          typeof payload.conversation_summary === "string" &&
          payload.conversation_summary.trim()
            ? payload.conversation_summary
            : typeof payload.response === "string" && payload.response.trim()
              ? payload.response
              : FRIENDLY_READY;
        break;
      case "clarification_required":
        assistantText =
          typeof payload.clarification?.question === "string" &&
          payload.clarification.question.trim()
            ? payload.clarification.question
            : FRIENDLY_CLARIFICATION;
        break;
      case "conflict":
        assistantText =
          typeof payload.response === "string" && payload.response.trim()
            ? payload.response
            : FRIENDLY_CONFLICT;
        break;
      case "error":
        assistantText = FRIENDLY_ERROR;
        break;
      default:
        console.error("AgentRec returned an unsupported status.", payload.status);
        assistantText = FRIENDLY_ERROR;
    }

    setLoading(false);
    updateStatusBadge(payload.status);
    renderShoppingPlan(payload);
    addMessage("agent", assistantText, { error: payload.status === "error" });
  } catch (error) {
    console.error("AgentRec request failed:", error);
    setLoading(false);
    updateStatusBadge(null);
    renderPlanEmpty(
      "Unable to update the plan",
      "The last request did not return a valid AgentRec response.",
    );
    addMessage("agent", FRIENDLY_ERROR, { error: true });
  } finally {
    messageInput.focus();
  }
}


// ============================================================
// 5. Keyboard, form, and example prompt interactions
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
