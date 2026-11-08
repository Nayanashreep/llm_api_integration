// Archie Chatbot Frontend Application
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const inspectBtn = document.getElementById("inspectBtn");
  const closeDrawerBtn = document.getElementById("closeDrawerBtn");
  const inspectorDrawer = document.getElementById("inspectorDrawer");
  const personaSelect = document.getElementById("personaSelect");
  const strategySelect = document.getElementById("strategySelect");
  const turnCounter = document.getElementById("turnCounter");
  const providerInfo = document.getElementById("providerInfo");
  const strategyInfo = document.getElementById("strategyInfo");

  // Error Banner Elements
  const errorBanner = document.getElementById("errorBanner");
  const errorTitle = document.getElementById("errorTitle");
  const errorMessage = document.getElementById("errorMessage");
  const errorSuggestion = document.getElementById("errorSuggestion");
  const dismissErrorBtn = document.getElementById("dismissErrorBtn");

  // Inspector Elements
  const metricTotalMsgs = document.getElementById("metricTotalMsgs");
  const metricActiveMsgs = document.getElementById("metricActiveMsgs");
  const metricEvictedMsgs = document.getElementById("metricEvictedMsgs");
  const metricStrategy = document.getElementById("metricStrategy");
  const inspectorSummaryBox = document.getElementById("inspectorSummaryBox");
  const inspectorPayloadBox = document.getElementById("inspectorPayloadBox");

  // State
  let sessionId = "session_" + Math.random().toString(36).substring(2, 9);
  let turn = 1;
  let isSending = false;
  let lastMetrics = null;

  // Auto-resize textarea
  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + "px";
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Starter cards
  document.querySelectorAll(".starter-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      messageInput.value = prompt;
      messageInput.focus();
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  // Load server config
  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      if (!res.ok) return;
      const data = await res.json();
      providerInfo.textContent = `Provider: ${data.provider.toUpperCase()} (${data.model})`;
      strategyInfo.textContent = `Strategy: ${data.active_strategy === "summarization" ? "Running Summarization" : "Sliding Window"} (Max ${data.max_history_messages} msgs)`;
      if (data.active_persona) {
        personaSelect.value = data.active_persona;
      }
      if (data.active_strategy) {
        strategySelect.value = data.active_strategy;
      }
    } catch (e) {
      console.warn("Could not fetch server config:", e);
    }
  }

  // Update server config on change
  async function updateConfig() {
    try {
      const payload = {
        persona: personaSelect.value,
        strategy: strategySelect.value,
      };
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      strategyInfo.textContent = `Strategy: ${strategySelect.value === "summarization" ? "Running Summarization" : "Sliding Window"}`;
    } catch (e) {
      console.warn("Could not update config:", e);
    }
  }

  personaSelect.addEventListener("change", updateConfig);
  strategySelect.addEventListener("change", updateConfig);

  // Inspector Drawer Toggle
  inspectBtn.addEventListener("click", async () => {
    inspectorDrawer.classList.toggle("hidden");
    if (!inspectorDrawer.classList.contains("hidden")) {
      await refreshInspector();
    }
  });

  closeDrawerBtn.addEventListener("click", () => {
    inspectorDrawer.classList.add("hidden");
  });

  async function refreshInspector() {
    try {
      const res = await fetch(`/api/history?session_id=${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      metricTotalMsgs.textContent = data.total_messages;
      metricStrategy.textContent = strategySelect.value === "summarization" ? "Summarization" : "Sliding Window";

      if (lastMetrics) {
        metricActiveMsgs.textContent = lastMetrics.active_window_messages || 0;
        metricEvictedMsgs.textContent = lastMetrics.evicted_messages || 0;
      } else {
        metricActiveMsgs.textContent = data.total_messages;
        metricEvictedMsgs.textContent = 0;
      }

      if (data.running_summary) {
        inspectorSummaryBox.textContent = data.running_summary;
      } else {
        inspectorSummaryBox.innerHTML = '<em class="dim">No evicted messages have been summarized yet.</em>';
      }

      // Display active messages payload
      const maxMsgs = 6;
      const activeMsgs = data.messages.slice(-maxMsgs);
      inspectorPayloadBox.textContent = JSON.stringify(activeMsgs, null, 2);
    } catch (e) {
      console.warn("Inspector update failed:", e);
    }
  }

  // Error banner controls
  function showError(title, msg, suggestion) {
    errorTitle.textContent = title || "Error";
    errorMessage.textContent = msg || "An unexpected error occurred.";
    errorSuggestion.textContent = suggestion ? `Tip: ${suggestion}` : "";
    errorBanner.classList.remove("hidden");
  }

  dismissErrorBtn.addEventListener("click", () => {
    errorBanner.classList.add("hidden");
  });

  // Reset conversation
  resetBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      // Clear UI
      messagesContainer.innerHTML = "";
      messagesContainer.appendChild(welcomeScreen);
      welcomeScreen.style.display = "flex";
      turn = 1;
      turnCounter.textContent = `Turn: 0`;
      lastMetrics = null;
      errorBanner.classList.add("hidden");
      await refreshInspector();
    } catch (e) {
      showError("Reset Failed", e.message);
    }
  });

  // Render message
  function appendMessage(role, text, metrics = null) {
    if (welcomeScreen.style.display !== "none") {
      welcomeScreen.style.display = "none";
    }

    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant" && typeof marked !== "undefined") {
      bubble.innerHTML = marked.parse(text);
    } else {
      bubble.textContent = text;
    }

    if (role === "assistant" && metrics) {
      const chip = document.createElement("div");
      chip.className = "meta-chip";
      chip.innerHTML = `
        <span class="chip-tag">${metrics.strategy}</span>
        <span>Active: ${metrics.active_window_messages}/${metrics.total_messages} msgs</span>
        ${metrics.evicted_messages > 0 ? `<span>• Evicted: ${metrics.evicted_messages}</span>` : ""}
        ${metrics.has_summary ? `<span class="chip-tag" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7;">+Memory Summary</span>` : ""}
      `;
      bubble.appendChild(chip);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Render typing indicator
  function createTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant typing-row";
    row.id = "typingRow";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble typing-indicator";
    bubble.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return row;
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typingRow");
    if (el) el.remove();
  }

  // Handle form submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isSending) return;

    const userText = messageInput.value.trim();
    if (!userText) return;

    // Reset input
    messageInput.value = "";
    messageInput.style.height = "auto";
    errorBanner.classList.add("hidden");

    // Display user message
    appendMessage("user", userText);
    turnCounter.textContent = `Turn: ${turn}`;

    isSending = true;
    sendBtn.disabled = true;
    const typingIndicator = createTypingIndicator();

    try {
      const payload = {
        message: userText,
        session_id: sessionId,
        strategy: strategySelect.value,
        persona: personaSelect.value,
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      removeTypingIndicator();
      const data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error_type || "API Error", data.message, data.suggestion);
        appendMessage("assistant", "⚠️ *I encountered an issue reaching the LLM service. Please check the error banner above for details.*");
      } else {
        lastMetrics = data.metrics;
        appendMessage("assistant", data.response, data.metrics);
        turn++;
        turnCounter.textContent = `Turn: ${turn - 1}`;
        if (!inspectorDrawer.classList.contains("hidden")) {
          await refreshInspector();
        }
      }
    } catch (err) {
      removeTypingIndicator();
      showError("Network Error", err.message, "Ensure server.py is running on localhost:8000.");
      appendMessage("assistant", "⚠️ *Failed to communicate with the local server.*");
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      messageInput.focus();
    }
  });

  // Initial load
  loadConfig();
});
