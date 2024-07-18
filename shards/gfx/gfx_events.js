var LibraryGFXEvents = {
  gfxClipboardGet__proxy: 'async',
  gfxClipboardGet: function (dataPtrPtr, readyPtr) {
    console.log("Reading data from clipboard");
    (async () => {
      let text = await navigator.clipboard.readText();
      Atomics.store(HEAP32, dataPtrPtr >> 2, stringToNewUTF8(text));
      Atomics.store(HEAP32, readyPtr >> 2, 0x01000000);
      console.log('Read data from clipboard: ', text);
    })();
  },
  gfxClipboardSet__proxy: 'async',
  gfxClipboardSet: function (dataCopy) {
    (async () => {
      let text = UTF8ToString(dataCopy);
      console.log('Copying data to clipboard: ', text);
      await navigator.clipboard.writeText(text);
      console.log('Data copied to clipboard');
      _free(dataCopy);
    })();
  },
  $gfxSetup__deps: ["$Browser"],
  $gfxSetup(canvasContainer, canvas) {
    const eh = this.getEventHandler();
    this.eventHandlerSetCanvas(eh, canvas.id, canvasContainer.id);

    canvasContainer.onmousemove = (e) => {
      e.type_ = 0;
      this.eventHandlerPostMouseEvent(eh, e);
    };
    const handleMouseEvent = (e) => {
      this.eventHandlerPostMouseEvent(eh, e);
    };
    canvasContainer.onmousedown = (e) => {
      e.type_ = 1;
      handleMouseEvent(e);
    };
    canvasContainer.onmouseup = (e) => {
      e.type_ = 2;
      handleMouseEvent(e);
    };
    window.oncontextmenu = (e) => {
      e.preventDefault();
    };

    var state = { cursorInPage: false };
    window.onmouseout = () => {
      state.cursorInPage = false;
    };
    window.onmouseover = () => {
      state.cursorInPage = true;
    };
    const trapKeyEvents = (code) => { return state.cursorInPage; };
    const handleKeyEvent = (event) => {
      event.domKey_ = event.keyCode;
      if (event.key.length == 1) {
        event.key_ = event.key.codePointAt(0);
      } else {
        event.key_ = 0;
      }

      this.eventHandlerPostKeyEvent(eh, event);
    };
    window.addEventListener('keydown', (event) => {
      if (trapKeyEvents(event.code)) {
        event.type_ = 0;
        handleKeyEvent(event);
        event.preventDefault();
      }
    }, true);
    window.addEventListener('keyup', (event) => {
      event.type_ = 1;
      handleKeyEvent(event);
      if (trapKeyEvents(event.code)) {
        event.preventDefault();
      }
    }, true);
    window.addEventListener('wheel', (event) => {
      console.log("Wheel", event);

      // Flip the wheel direction to translate from browser wheel direction
      // (+:down) to SDL direction (+:up)
      var deltaY = -Browser.getMouseWheelDelta(event);
      // Quantize to integer so that minimum scroll is at least +/- 1.
      deltaY = (deltaY == 0) ? 0 : (deltaY > 0 ? Math.max(deltaY, 1) : Math.min(deltaY, -1));

      this.eventHandlerPostWheelEvent(eh, { deltaY: deltaY });
    });

    (async function resizeCanvasLoop() {
      var lastW, lastH;
      var lastPixelRatio;
      while (true) {
        const rect = canvasContainer.getBoundingClientRect();
        const pixelRatio = window.devicePixelRatio;
        if (lastW !== rect.width || lastH !== rect.height || lastPixelRatio !== pixelRatio) {
          let canvasWidth = rect.width * pixelRatio;
          let canvasHeight = rect.height * pixelRatio;
          this.eventHandlerPostDisplayFormat(eh, rect.width, rect.height, canvasWidth, canvasHeight, pixelRatio);
          lastW = rect.width;
          lastH = rect.height;
          lastPixelRatio = pixelRatio;
        }
        await new Promise(done => setTimeout(done, 1));
      }
    }.bind(this))();
  }
};

addToLibrary(LibraryGFXEvents);
