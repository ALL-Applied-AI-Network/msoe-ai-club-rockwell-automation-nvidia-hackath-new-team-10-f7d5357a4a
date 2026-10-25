"use strict";

const POLL_INTERVAL_MS = 1000;

const STATE_SYMBOLS = {
  PASS: "✓",
  DEFECT: "✕",
  MISSED: "!",
  UNSURE: "?",
  WAITING: "○",
};

const STATE_PRIORITY = {
  DEFECT: 0,
  MISSED: 1,
  UNSURE: 2,
  PASS: 3,
  WAITING: 4,
};

let lastEventKey = "";
let selectedPosition = null;
let latestHoles = new Map();

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const element = byId(id);
  if (element) {
    element.textContent = value;
  }
}

function positionNumber(position) {
  return Number(position.slice(1));
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function severityMessage(severity) {
  const messages = {
    ACTION_REQUIRED:
      "Action required — review defect or missed positions",
    REVIEW_REQUIRED:
      "Review required — one or more positions are unsure",
    NORMAL:
      "Inspection normal",
    DATA_UNAVAILABLE:
      "Inspection data unavailable — check the checker process",
    WAITING_FOR_EVENT:
      "Waiting for a complete inspection event",
  };

  return messages[severity] || severity;
}

function deriveSeverity(event, health) {
  if (event) {
    return event.severity || "WAITING_FOR_EVENT";
  }

  if (
    health.state === "UNAVAILABLE" ||
    health.state === "STALE" ||
    health.state === "DISCONNECTED"
  ) {
    return "DATA_UNAVAILABLE";
  }

  return "WAITING_FOR_EVENT";
}

function renderBanner(event, health) {
  const banner = byId("event-banner");
  if (!banner) {
    return;
  }

  const severity = deriveSeverity(event, health);

  banner.className =
    `event-banner severity-${severity.toLowerCase()}`;

  banner.textContent = severityMessage(severity);

  if (!event && health.message) {
    banner.textContent += ` · ${health.message}`;
  }
}

function renderHealth(health) {
  const state = health.state || "UNKNOWN";
  const status = byId("source-status");

  if (status) {
    status.textContent = state;
    status.className =
      `status-label health-${state.toLowerCase()}`;
  }

  if (health.source_age_seconds !== null &&
      health.source_age_seconds !== undefined) {
    setText(
      "updated-label",
      `Source age: ${formatNumber(
        health.source_age_seconds,
        1
      )} s`
    );
  } else {
    setText("updated-label", "Source age: —");
  }
}

function renderCounts(counts) {
  const states = [
    "pass",
    "defect",
    "missed",
    "unsure",
    "waiting",
  ];

  for (const state of states) {
    setText(
      `count-${state}`,
      counts[state.toUpperCase()] ?? 0
    );
  }
}

/*
 * Physical layout:
 *
 * G26 G21 G16 G11 G06 G01
 * G27 G22 G17 G12 G07 G02
 * G28 G23 G18 G13 G08 G03
 * G29 G24 G19 G14 G09 G04
 * G30 G25 G20 G15 G10 G05
 */
function fixtureOrder(holes) {
  return [...holes].sort((left, right) => {
    const leftIndex = positionNumber(left.position);
    const rightIndex = positionNumber(right.position);

    const leftRow = (leftIndex - 1) % 5;
    const rightRow = (rightIndex - 1) % 5;

    const leftColumn =
      5 - Math.floor((leftIndex - 1) / 5);
    const rightColumn =
      5 - Math.floor((rightIndex - 1) / 5);

    return (
      leftRow * 6 +
      leftColumn -
      (rightRow * 6 + rightColumn)
    );
  });
}

function createHoleButton(hole) {
  const button = document.createElement("button");

  button.type = "button";
  button.className =
    `hole state-${hole.final_state.toLowerCase()}`;

  button.setAttribute(
    "aria-label",
    `${hole.position}: ${hole.final_state}`
  );

  const position = document.createElement("span");
  position.className = "hole-position";
  position.textContent = hole.position;

  const result = document.createElement("strong");
  result.textContent =
    `${STATE_SYMBOLS[hole.final_state] || ""} ` +
    hole.final_state;

  const visual = document.createElement("span");
  visual.className = "hole-visual";
  visual.textContent = hole.predicted_state || "—";

  button.append(position, result, visual);

  button.addEventListener("click", () => {
    showPositionDetails(hole.position);
  });

  return button;
}

function renderFixture(holes) {
  const grid = byId("fixture-grid");
  if (!grid) {
    return;
  }

  latestHoles = new Map(
    holes.map((hole) => [hole.position, hole])
  );

  grid.replaceChildren();

  for (const hole of fixtureOrder(holes)) {
    grid.append(createHoleButton(hole));
  }
}

function flaggedPositions(holes) {
  return holes
    .filter((hole) =>
      ["DEFECT", "MISSED", "UNSURE"].includes(
        hole.final_state
      )
    )
    .sort((left, right) => {
      const priorityDifference =
        STATE_PRIORITY[left.final_state] -
        STATE_PRIORITY[right.final_state];

      if (priorityDifference !== 0) {
        return priorityDifference;
      }

      return (
        positionNumber(left.position) -
        positionNumber(right.position)
      );
    });
}

function renderReviewList(holes) {
  const list = byId("review-list");
  if (!list) {
    return;
  }

  list.replaceChildren();

  const flagged = flaggedPositions(holes);

  if (flagged.length === 0) {
    const message = document.createElement("p");
    message.className = "all-clear";
    message.textContent =
      "✓ No positions require operator review.";
    list.append(message);
    return;
  }

  for (const hole of flagged) {
    const button = document.createElement("button");

    button.type = "button";
    button.className =
      `review-item state-${hole.final_state.toLowerCase()}`;

    button.textContent =
      `${STATE_SYMBOLS[hole.final_state]} ` +
      `${hole.position} — ${hole.final_state}`;

    if (
      hole.final_state === "DEFECT" &&
      hole.predicted_state
    ) {
      button.textContent +=
        ` (${hole.predicted_state})`;
    }

    button.addEventListener("click", () => {
      showPositionDetails(hole.position);
    });

    list.append(button);
  }
}

function showPositionDetails(position) {
  const hole = latestHoles.get(position);
  if (!hole) {
    return;
  }

  selectedPosition = position;

  const values = [
    hole.position,
    hole.final_state,
    hole.predicted_state || "—",
    formatNumber(hole.score),
    hole.reason || "No reason supplied",
  ];

  const details = byId("position-details");
  if (!details) {
    return;
  }

  const fields = details.querySelectorAll("dd");

  fields.forEach((field, index) => {
    field.textContent = values[index] ?? "—";
  });

  document.querySelectorAll(".hole").forEach((button) => {
    button.classList.toggle(
      "selected",
      button.getAttribute("aria-label")
        ?.startsWith(`${position}:`)
    );
  });
}

function chooseDefaultPosition(holes) {
  if (
    selectedPosition &&
    holes.some(
      (hole) => hole.position === selectedPosition
    )
  ) {
    return selectedPosition;
  }

  const flagged = flaggedPositions(holes);

  if (flagged.length > 0) {
    return flagged[0].position;
  }

  return holes[0]?.position || null;
}

function updateOverlay(eventKey) {
  const overlay = byId("overlay");
  const placeholder = byId("image-placeholder");

  if (!overlay) {
    return;
  }

  const candidate = new Image();

  candidate.onload = () => {
    overlay.src = candidate.src;
    overlay.hidden = false;

    if (placeholder) {
      placeholder.hidden = true;
    }
  };

  /*
   * If loading fails, retain the previous overlay. This is
   * important while the checker atomically replaces the file.
   */
  candidate.src =
    `/overlay?event=${encodeURIComponent(eventKey)}` +
    `&t=${Date.now()}`;
}

function renderNewEvent(payload) {
  const event = payload.event;
  const holes = payload.holes || [];

  if (!event || holes.length !== 30) {
    return;
  }

  const eventKey = `${event.run}:${event.event}`;

  setText(
    "event-label",
    `Run ${event.run} · Event ${event.event}`
  );

  setText(
    "frontier-label",
    `Fill frontier: ${event.fill_frontier ?? "—"}`
  );

  renderCounts(payload.counts || {});
  renderFixture(holes);
  renderReviewList(holes);

  const defaultPosition = chooseDefaultPosition(holes);
  if (defaultPosition) {
    showPositionDetails(defaultPosition);
  }

  lastEventKey = eventKey;
}

function renderSnapshot(payload) {
  const health = payload.health || {
    state: "UNKNOWN",
    message: "",
  };

  renderHealth(health);
  renderBanner(payload.event, health);

  /*
   * Retain the previous event if the source becomes temporarily
   * unavailable or no complete event has been produced yet.
   */
  if (!payload.event) {
    return;
  }

  const eventKey =
    `${payload.event.run}:${payload.event.event}`;

  /*
   * Event contents are immutable after all 30 rows are written,
   * so the grid only needs rebuilding when the event changes.
   */
  if (eventKey !== lastEventKey) {
    renderNewEvent(payload);
  }

  /*
   * The checker writes the 30 result rows before replacing
   * overlay_latest.png. Refresh every poll so an initial request
   * that receives the previous image self-corrects.
   */
  if (eventKey === lastEventKey) {
    updateOverlay(eventKey);
  }
}

async function loadSession() {
  try {
    const response = await fetch(
      "/api/v1/session",
      { cache: "no-store" }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const session = await response.json();

    const inspectorName =
      session.inspector?.name
        ? ` · ${session.inspector.name}`
        : "";

    setText(
      "session-label",
      `${session.session || "Unknown session"}` +
      inspectorName
    );
  } catch (error) {
    setText(
      "session-label",
      "Session metadata unavailable"
    );
  }
}

function showDisconnected(error) {
  const health = {
    state: "DISCONNECTED",
    message: String(error),
  };

  renderHealth(health);
  renderBanner(null, health);

  /*
   * Do not clear the last complete event or image. Operators
   * should still be able to see the last known result.
   */
}

async function pollSnapshot() {
  try {
    const response = await fetch(
      "/api/v1/snapshot",
      { cache: "no-store" }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    renderSnapshot(payload);
  } catch (error) {
    showDisconnected(error);
  } finally {
    window.setTimeout(
      pollSnapshot,
      POLL_INTERVAL_MS
    );
  }
}

loadSession();
pollSnapshot();