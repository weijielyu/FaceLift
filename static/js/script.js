// For RGB / global segmentation results
var video_width = 779;
const VIDEO_ASPECT_RATIO = 779 / 518;
var display_level = 0;
var video_names = ["teaser"];
    
var videos = [];
var current_video_idx = 0;

function load_videos() {
  for (var i = 0; i < video_names.length; i++) {
    videos.push(document.getElementById(video_names[i])); 
  }
}

window.onload = function() {
  resize_canvas();
  load_videos();
  videos[0].play();
}

/* Synchronize main_results, and its canvas(es) to have the same size. */
function resize_canvas() {
  var main_results = document.getElementById('image-compare-canvas');
  var width = main_results.offsetWidth;

  var height = width / VIDEO_ASPECT_RATIO;
  main_results.height = height;
  main_results.style.height = height;

  video_width = width;

  var canvas = document.getElementById('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = width;
  canvas.style.height = height;
}

// Need to trigger a `resize` when window size changes. 
// In particular, need to do resize after content loaded, to define height of the canvas!
// Otherwise, the canvas for main-results display won't work.
window.addEventListener('resize', resize_canvas, false);
document.addEventListener("DOMContentLoaded", function() { resize_canvas(); });

/* Image compare utility. Requires jquery + tabler-icons. */
$(() => {
  $(".image-compare").each((_index, parent) => {
    const $parent = $(parent);
    const before = $parent.data("before-label") || "Before";
    const after = $parent.data("after-label") || "After";
    $parent.append(
      "<div id='image-compare-handle' class='image-compare-handle'><i class='ti ti-arrows-horizontal'></i></div>" +
        "<div id='image-compare-before' class='image-compare-before'><div>" +
        before +
        "</div></div>" +
        "<div id='image-compare-after' class='image-compare-after'><div>" +
        after +
        "</div></div>",
    );
  });

  setInterval(() => {
    $(".image-compare").each((_index, parent) => {
      const $parent = $(parent);
      const $handle = $parent.children(".image-compare-handle");

      const currentLeft = $handle.position().left;

      // Linear dynamics + PD controller : - )
      const Kp = 0.03;
      const Kd = 0.2;

      let velocity = $parent.data("velocity") || 0;
      let targetLeft = $parent.data("targetX");
      if (targetLeft !== undefined) {
        const padding = 10;
        const parentWidth = $parent.width();
        if (targetLeft <= padding) targetLeft = 0;
        if (targetLeft >= parentWidth - padding) targetLeft = parentWidth;

        const delta = targetLeft - currentLeft;
        velocity += Kp * delta;
      }
      velocity -= Kd * velocity;

      // Update velocity.
      $parent.data("velocity", velocity);

      const newLeft = currentLeft + velocity;
      $parent.children(".image-compare-handle").css("left", newLeft + "px");
      $parent.children(".image-compare-before").width(newLeft + "px");
      // $parent.children("img:not(:first-child)").width(newLeft + "px");

      // $parent.children(".image-compare-after").style.right = 0;
      $parent.children(".image-compare-after").css("left", newLeft + "px");
      $parent.children(".image-compare-after").width(video_width - newLeft + "px");

      var canvas = document.getElementById('canvas');
      var ctx = canvas.getContext('2d');

      if (videos.length == 0) load_videos();

      const newLeftVideo = newLeft * 779 / video_width;
      video = videos[current_video_idx];

      // drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
      ctx.drawImage(
        video, 
        0, 0, newLeftVideo, 518,
        0, 0, newLeft, video_width/VIDEO_ASPECT_RATIO
        );  // RGB
      ctx.drawImage(
        video,
        779*(display_level+1)+newLeftVideo, 0, 779-newLeftVideo, 518, newLeft, 
        0, video_width-newLeft, video_width/VIDEO_ASPECT_RATIO
        ); // Segmentation
    });
  }, 10);

  $(".image-compare").bind("mousedown touchstart", (evt) => {
    const $parent = $(evt.target.closest(".image-compare"));
    $parent.data("dragging", true);

    if (evt.type == "mousedown")
      $parent.data("targetX", evt.pageX - $parent.offset().left);
    else if (evt.type == "touchstart")
      $parent.data("targetX", evt.touches[0].pageX - $parent.offset().left);
  });

  $(document)
    .bind("mouseup touchend", () => {
      $(".image-compare").each((_index, parent) => {
        $(parent).data("dragging", false);
      });
    })
    .bind("mousemove touchmove", (evt) => {
      $(".image-compare").each((_index, parent) => {
        const $parent = $(parent);
        if (!$parent.data("dragging")) return;

        if (evt.type == "mousemove")
          $parent.data("targetX", evt.pageX - $parent.offset().left);
        else if (evt.type == "touchmove")
          $parent.data("targetX", evt.touches[0].pageX - $parent.offset().left);
      });
    });
}, 1000 / 60);  // 30fps

/* Switcher. */
// Contains logic for switching between coarse/medium/fine.
$(() => {
  $(".switcher").each((switcher_index, switcher) => {
    const $switcher = $(switcher);

    const $inputContainer = $("<div>", { class: "switcher-labels" });

    let $current = null;

    $switcher.children().each((switcher_child_index, child) => {
      const $child = $(child);

      const linkId =
        "switcher-group-" +
        switcher_index.toString() +
        "-" +
        switcher_child_index.toString();
      const $input = $("<input>", {
        type: "radio",
        name: "switcher-group-" + switcher_index.toString(),
        id: linkId,
        checked: switcher_child_index === 0,
        click: function () {
          // Your onclick event logic goes here
          $current.addClass("switcher-hidden");
          display_level = switcher_child_index;

          $current = $([]);
          $.merge($current, $child);
          $.merge($current, $input);
          $.merge($current, $label);

          $current.removeClass("switcher-hidden");
        },
      });
      const $label = $("<label>", {
        text: $child.data("switcher-label"),
        for: linkId,
      });
      $inputContainer.append($("<div>").append($input).append($label));

      if (switcher_child_index !== 0) {
        $child.addClass("switcher-hidden");
        $input.addClass("switcher-hidden");
        $label.addClass("switcher-hidden");
      } else {
        $current = $([]);
        $.merge($current, $child);
        $.merge($current, $input);
        $.merge($current, $label);
      }
    });

    const $label = $("<label>", {
      text: $switcher.data("switcher-title") + ":",
    });
    $inputContainer.prepend($label);

    $switcher.append($inputContainer);
  });
});

/* Switcher. */
// Contains logic for switching between coarse/medium/fine.
$(() => {
  $(".results-slide-row").each((switcher_index, switcher) => {
    const $switcher = $(switcher);
    console.log($switcher);
    console.log($switcher.children());

    $switcher.children().each((switcher_child_index, child) => {
      const $child = $(child);

      const $input = $("<button>", {
        class: "thumbnail-btn",
        id: "thumb-" + switcher_index.toString(),
        click: function () {
          // Your onclick event logic goes here
          current_video_idx = switcher_child_index;
          current_video = videos[current_video_idx]
          current_video.currentTime = 0;
          current_video.play();
          set_play_pause_icon();
        },
      });
      const $img = $("<img>", {
        class: "thumbnails",
        alt: "paper",
        src: $child.data("img-src"),
      });
      $input.append($img);
      const $label = $("<label>", {
        text: $child.data("label"),
        class: "thumbnail_label",
      });
      $input.append($label);
      $switcher.append($input);
    });
  });
});

function set_play_pause_icon() {
  button = document.getElementById('play-btn')
  current_video = videos[current_video_idx]
  if (current_video.paused) {
    button.classList.remove("fa-pause");
    button.classList.add("fa-play");
  } else {
    button.classList.add("fa-pause");
    button.classList.remove("fa-play");
  }
}

function play_pause() {
  current_video = videos[current_video_idx]
  if (current_video.paused) {
    current_video.play();
  } else {
    current_video.pause();
  }
  set_play_pause_icon();
}

function fullscreen() {
  current_video = videos[current_video_idx]
  current_video.style.visibility = "visible";
  const fullscreenElement =
    document.fullscreenElement ||
    document.mozFullScreenElement ||
    document.webkitFullscreenElement ||
    document.msFullscreenElement;
  if (fullscreenElement) {
    exitFullscreen();
  } else {
    launchIntoFullscreen(current_video);
  }
}

function download() {
  current_video = videos[current_video_idx]
  var link = document.createElement('a');
  link.download = video_names[current_video_idx] + '.mp4';
  link.href = download_paths[current_video_idx];
  link.click();
}

function launchIntoFullscreen(element) {
  if (element.requestFullscreen) {
    element.requestFullscreen();
  } else if (element.mozRequestFullScreen) {
    element.mozRequestFullScreen();
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen();
  } else {
    element.classList.toggle('fullscreen');
  }
}

function exitFullscreen() {
  if (document.exitFullscreen) {
    document.exitFullscreen();
  } else if (document.mozCancelFullScreen) {
    document.mozCancelFullScreen();
  } else if (document.webkitExitFullscreen) {
    document.webkitExitFullscreen();
  }
}

if (document.addEventListener)
{
 document.addEventListener('fullscreenchange', exitHandler, false);
 document.addEventListener('mozfullscreenchange', exitHandler, false);
 document.addEventListener('MSFullscreenChange', exitHandler, false);
 document.addEventListener('webkitfullscreenchange', exitHandler, false);
}

function exitHandler()
{
 if (!document.webkitIsFullScreen && !document.mozFullScreen && !document.msFullscreenElement)
 {
  current_video = videos[current_video_idx]
  current_video.style.visibility = "hidden";
 }
}