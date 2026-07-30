/**
 * Work out what a capture actually is, so nobody has to declare it.
 *
 * The ingestion route already distinguishes a single source from a batch, an
 * upload from a link, and a signed-in capture from a public one. It just needed
 * the form to say which. This reads the one visible field and fills the fields
 * the route expects; every name here already existed.
 */
(function () {
  "use strict";

  var form = document.getElementById("capture-form");
  if (!form) return;

  var input = document.getElementById("capture-input");
  var batch = document.getElementById("capture-batch");
  var mode = document.getElementById("capture-mode");
  var dropzone = document.getElementById("dropzone");
  var picker = document.getElementById("capture-picker");
  var knowledge = document.getElementById("capture-knowledge");
  var list = document.getElementById("capture-list");
  var bookmarks = document.getElementById("capture-bookmarks");
  var bookmarksForm = document.getElementById("bookmarks-form");
  var detect = document.getElementById("capture-detect");
  var detectMain = document.getElementById("capture-detect-main");
  var detectSub = document.getElementById("capture-detect-sub");
  var clear = document.getElementById("capture-clear");
  var submit = document.getElementById("capture-submit");
  var deep = document.getElementById("depth-deep");
  var deepNote = document.getElementById("depth-deep-note");
  var deepOption = document.getElementById("depth-deep-option");
  var consentUpload = document.getElementById("consent-upload");
  var consentUploadText = document.getElementById("consent-upload-text");
  var consentBrowser = document.getElementById("consent-browser");
  var authorizeUpload = document.getElementById("authorize-upload");
  var authorizeBrowser = document.getElementById("authorize-browser");

  var LIST_TYPES = [".txt", ".csv", ".yaml", ".yml"];
  var deepNoteDefault = deepNote ? deepNote.textContent.trim() : "";
  var staged = null;

  function extensionOf(name) {
    var dot = name.lastIndexOf(".");
    return dot === -1 ? "" : name.slice(dot).toLowerCase();
  }

  /**
   * A dropped file goes to whichever input the route reads for that kind.
   * `.json` is treated as a bookmarks export because that is what the
   * bookmarklet writes; the chip says so, so a mistake is visible before
   * anything is sent.
   */
  function routeFor(name) {
    var extension = extensionOf(name);
    if (extension === ".json") return "bookmarks";
    if (LIST_TYPES.indexOf(extension) !== -1) return "list";
    return "document";
  }

  function readableSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function hostOf(value) {
    try {
      return new URL(value).hostname.replace(/^www\./, "");
    } catch (error) {
      return "link";
    }
  }

  /** Move a file into the input whose name the route reads. */
  function assign(target, file) {
    var transfer = new DataTransfer();
    if (file) transfer.items.add(file);
    target.files = transfer.files;
  }

  function clearFiles() {
    [knowledge, list, bookmarks, picker].forEach(function (field) {
      if (field) assign(field, null);
    });
  }

  function lines() {
    return input.value
      .split("\n")
      .map(function (line) { return line.trim(); })
      .filter(Boolean);
  }

  function reading() {
    if (staged) {
      var route = staged.route;
      return {
        main: route === "bookmarks" ? "Bookmarks export" : route === "list" ? "Link list" : "File",
        sub: staged.file.name + " · " + readableSize(staged.file.size),
        action: route === "bookmarks" ? "Import bookmarks" : "Capture file",
        deepable: false,
        file: true
      };
    }
    var found = lines();
    var urls = found.filter(function (line) { return /^https?:\/\//i.test(line); });
    if (!found.length) {
      return { main: "Nothing yet", sub: "the field is empty", action: "Capture", deepable: false, idle: true };
    }
    if (urls.length === found.length && urls.length === 1) {
      return { main: "1 link", sub: hostOf(urls[0]), action: "Capture 1 source", deepable: true };
    }
    if (urls.length === found.length) {
      return {
        main: urls.length + " links",
        sub: "captured as a batch",
        action: "Capture " + urls.length + " sources",
        deepable: false,
        many: true
      };
    }
    if (!urls.length) {
      return {
        main: "Pasted text",
        sub: input.value.trim().length.toLocaleString() + " characters",
        action: "Capture 1 source",
        deepable: false
      };
    }
    return {
      main: urls.length + " links",
      sub: "with " + (found.length - urls.length) + " lines of text, captured as a batch",
      action: "Capture " + urls.length + " sources",
      deepable: false,
      many: true
    };
  }

  function render() {
    var state = reading();

    // Deep capture reads one post's thread, so it only applies to a single link.
    if (deep && !deep.disabled) {
      var blocked = !state.deepable && !state.idle;
      if (blocked && deep.checked) deep.checked = false;
      deep.disabled = blocked;
      if (deepOption) deepOption.classList.toggle("is-unavailable", blocked);
      if (deepNote) {
        deepNote.textContent = blocked
          ? state.many
            ? "One link at a time."
            : "Needs a link."
          : deepNoteDefault;
      }
    }

    var wantsBrowser = Boolean(deep && deep.checked && !deep.disabled);
    if (consentBrowser) {
      consentBrowser.hidden = !wantsBrowser;
      if (!wantsBrowser && authorizeBrowser) authorizeBrowser.checked = false;
    }
    var wantsUpload = Boolean(staged && staged.route === "document");
    if (consentUpload) {
      consentUpload.hidden = !wantsUpload;
      if (!wantsUpload && authorizeUpload) authorizeUpload.checked = false;
      if (wantsUpload && consentUploadText) {
        consentUploadText.textContent =
          "Let STEERING read " + staged.file.name +
          ". It is processed in memory and the binary is discarded.";
      }
    }

    detect.classList.toggle("is-idle", Boolean(state.idle));
    detectMain.textContent = state.main;
    detectSub.textContent = state.sub;
    submit.textContent = wantsBrowser ? "Capture with the browser" : state.action;
    if (clear) clear.hidden = !staged;
    input.disabled = Boolean(staged);
  }

  function stage(file) {
    if (!file) return;
    staged = { file: file, route: routeFor(file.name) };
    clearFiles();
    if (staged.route === "bookmarks") assign(bookmarks, file);
    else if (staged.route === "list") assign(list, file);
    else assign(knowledge, file);
    render();
  }

  function unstage() {
    staged = null;
    clearFiles();
    render();
  }

  input.addEventListener("input", render);
  form.addEventListener("change", function (event) {
    if (event.target.name === "depth") render();
  });
  if (clear) clear.addEventListener("click", unstage);
  if (picker) {
    document.getElementById("capture-browse").addEventListener("click", function () {
      picker.click();
    });
    picker.addEventListener("change", function () {
      if (picker.files && picker.files[0]) stage(picker.files[0]);
    });
  }

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-holding");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      if (name === "dragleave" && dropzone.contains(event.relatedTarget)) return;
      dropzone.classList.remove("is-holding");
    });
  });
  dropzone.addEventListener("drop", function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files[0]) stage(files[0]);
  });

  input.addEventListener("keydown", function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  /**
   * Fill the route's own fields at the last moment, from what is on screen.
   * A batch is carried in `batch` with mode `batch`; a single source stays in
   * `source`; deep capture is mode `browser`, which the route reads as one URL.
   */
  form.addEventListener("submit", function (event) {
    if (staged && staged.route === "bookmarks") {
      event.preventDefault();
      if (window.htmx) window.htmx.trigger(bookmarksForm, "submit");
      else bookmarksForm.submit();
      return;
    }
    var state = reading();
    if (staged) {
      // The route reads an upload before anything else; a list needs batch mode.
      mode.value = staged.route === "list" ? "batch" : "url";
      batch.value = "";
      return;
    }
    if (deep && deep.checked && !deep.disabled) {
      mode.value = "browser";
      batch.value = "";
      return;
    }
    if (state.many) {
      mode.value = "batch";
      batch.value = input.value;
      return;
    }
    mode.value = "url";
    batch.value = "";
  });

  render();
})();
