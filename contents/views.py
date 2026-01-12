import os
import re
import mimetypes

from io import BytesIO
from PIL import Image as PILImage

from django.contrib import messages
from django.core.files.base import ContentFile
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import CreateView, DeleteView, DetailView, ListView

from .forms import VideoUploadForm, ImageUploadForm
from .models import Video, Image as ImageModel


# ============ 통합 미디어 목록 ============
class ContentListView(ListView):
    """동영상과 이미지 통합 목록"""

    template_name = "contents/content_list.html"
    context_object_name = "content_items"
    paginate_by = 12

    def get_queryset(self):
        tab = self.request.GET.get("tab", "all")
        search = self.request.GET.get("search", "")
        sort = self.request.GET.get("sort", "newest")
        
        content_items = []

        # 동영상 가져오기
        if tab in ["all", "video"]:
            videos = Video.objects.all()
            if search:
                videos = videos.filter(title__icontains=search)

            for video in videos:
                content_items.append(
                    {
                        "type": "video",
                        "id": video.id,
                        "title": video.title,
                        "description": video.description,
                        "file": video.file,
                        "thumbnail": video.thumbnail,
                        "file_size": video.get_file_size_display(),
                        "uploaded_at": video.uploaded_at,
                        "obj": video,
                    }
                )

        # 이미지 가져오기
        if tab in ["all", "image"]:
            images = ImageModel.objects.all()
            if search:
                images = images.filter(title__icontains=search)

            for image in images:
                content_items.append(
                    {
                        "type": "image",
                        "id": image.id,
                        "title": image.title,
                        "description": image.description,
                        "file": image.file,
                        "thumbnail": image.file,
                        "file_size": image.get_file_size_display(),
                        "uploaded_at": image.uploaded_at,
                        "resolution": image.get_resolution_display(),
                        "obj": image,
                    }
                )

        # 정렬 적용
        if sort == "oldest":
            content_items.sort(key=lambda x: x["uploaded_at"])
        elif sort == "name_asc":
            content_items.sort(key=lambda x: x["title"].lower())
        elif sort == "name_desc":
            content_items.sort(key=lambda x: x["title"].lower(), reverse=True)
        else:  # newest (기본값)
            content_items.sort(key=lambda x: x["uploaded_at"], reverse=True)

        return content_items

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["tab"] = self.request.GET.get("tab", "all")
        context["search"] = self.request.GET.get("search", "")
        context["sort"] = self.request.GET.get("sort", "newest") 
        context["video_count"] = Video.objects.count()
        context["image_count"] = ImageModel.objects.count()
        return context


# ============ 동영상 뷰 ============
class VideoCreateView(CreateView):
    model = Video
    form_class = VideoUploadForm
    template_name = "contents/video_upload.html"

    def form_valid(self, form):
        # 1단계: 동영상 파일 먼저 저장
        self.object = form.save(commit=False)
        self.object.file_size = self.object.file.size if self.object.file else 0
        self.object.save()
        
        print(f"\n{'='*60}")
        print(f"📹 동영상 업로드: {self.object.title}")
        print(f"{'='*60}")

        # 2단계: 썸네일 생성
        if self.object.file:
            try:
                video_path = self.object.file.path
                print(f"📂 동영상 경로: {video_path}")
                print(f"✅ 파일 존재: {os.path.exists(video_path)}")
                
                # 썸네일 생성
                thumbnail_content = generate_thumbnail_from_video(video_path)
                
                if thumbnail_content:
                    # 파일명 생성
                    original_name = os.path.splitext(os.path.basename(self.object.file.name))[0]
                    thumbnail_filename = f"{original_name}_thumb.jpg"
                    
                    print(f"📝 썸네일 파일명: {thumbnail_filename}")
                    
                    # 썸네일 저장 (save=True로 즉시 저장)
                    self.object.thumbnail.save(
                        thumbnail_filename,
                        thumbnail_content,
                        save=True  # ⭐ 즉시 저장
                    )
                    
                    print(f"✅ 썸네일 저장 성공!")
                    print(f"📍 썸네일 경로: {self.object.thumbnail.path}")
                    print(f"📍 썸네일 URL: {self.object.thumbnail.url}")
                else:
                    print(f"⚠️ 썸네일 생성 실패 - generate_thumbnail 반환값 None")
                    
            except Exception as e:
                print(f"❌ 썸네일 생성 중 오류: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"{'='*60}\n")

        messages.success(self.request, "동영상이 성공적으로 업로드되었습니다!")
        return redirect(self.get_success_url())

    def form_invalid(self, form):
        messages.error(
            self.request, "업로드 중 오류가 발생했습니다. 다시 시도해주세요."
        )
        return super().form_invalid(form)

    def get_success_url(self):
        return reverse("video_detail", kwargs={"pk": self.object.pk})


class VideoDetailView(DetailView):
    model = Video
    template_name = "contents/video_detail.html"
    context_object_name = "video"


class VideoDeleteView(DeleteView):
    model = Video
    template_name = "contents/video_delete.html"
    context_object_name = "video"
    success_url = reverse_lazy("content_list")

    def delete(self, request, *args, **kwargs):
        # Signal이 파일 삭제를 자동으로 처리
        messages.success(request, "동영상이 삭제되었습니다.")
        return super().delete(request, *args, **kwargs)


class VideoStreamView(View):
    def get(self, request, pk):
        video = get_object_or_404(Video, pk=pk)
        video_path = video.file.path

        if not os.path.exists(video_path):
            raise Http404("동영상 파일을 찾을 수 없습니다.")

        file_size = os.path.getsize(video_path)
        content_type, _ = mimetypes.guess_type(video_path)
        if not content_type:
            content_type = "video/mp4"

        range_header = request.META.get("HTTP_RANGE", "").strip()
        range_re = re.compile(r"bytes\s*=\s*(\d+)\s*-\s*(\d*)", re.I)
        range_match = range_re.match(range_header)

        if range_match:
            first_byte, last_byte = range_match.groups()
            first_byte = int(first_byte) if first_byte else 0
            last_byte = int(last_byte) if last_byte else file_size - 1

            if last_byte >= file_size:
                last_byte = file_size - 1

            length = last_byte - first_byte + 1

            with open(video_path, "rb") as file:
                file.seek(first_byte)
                data = file.read(length)

            response = HttpResponse(data, status=206, content_type=content_type)
            response["Content-Length"] = str(length)
            response["Content-Range"] = f"bytes {first_byte}-{last_byte}/{file_size}"
            response["Accept-Ranges"] = "bytes"
            return response

        response = FileResponse(open(video_path, "rb"), content_type=content_type)
        response["Content-Length"] = str(file_size)
        response["Accept-Ranges"] = "bytes"
        return response


# ============ 이미지 뷰 ============
class ImageCreateView(CreateView):
    model = ImageModel
    form_class = ImageUploadForm
    template_name = "contents/image_upload.html"

    def form_valid(self, form):
        self.object = form.save(commit=False)

        if self.object.file:
            self.object.file_size = self.object.file.size

            # 이미지 해상도 저장
            try:
                img = PILImage.open(self.object.file)
                self.object.width, self.object.height = img.size
            except Exception as e:
                print(f"이미지 해상도 추출 실패: {e}")

        self.object.save()
        messages.success(self.request, "이미지가 성공적으로 업로드되었습니다!")
        return redirect(self.get_success_url())

    def form_invalid(self, form):
        messages.error(
            self.request, "업로드 중 오류가 발생했습니다. 다시 시도해주세요."
        )
        return super().form_invalid(form)

    def get_success_url(self):
        return reverse("image_detail", kwargs={"pk": self.object.pk})


class ImageDetailView(DetailView):
    model = ImageModel
    template_name = "contents/image_detail.html"
    context_object_name = "image"


class ImageDeleteView(DeleteView):
    model = ImageModel
    template_name = "contents/image_delete.html"
    context_object_name = "image"
    success_url = reverse_lazy("content_list")

    def delete(self, request, *args, **kwargs):
        # Signal이 파일 삭제를 자동으로 처리
        messages.success(request, "이미지가 삭제되었습니다.")
        return super().delete(request, *args, **kwargs)


# ============ 통합 업로드 뷰 ============
class ContentUploadView(View):
    """동영상 또는 이미지 업로드"""

    def get(self, request):
        upload_type = request.GET.get("type", "video")

        if upload_type == "image":
            return ImageCreateView.as_view()(request)
        else:
            return VideoCreateView.as_view()(request)

    def post(self, request):
        upload_type = request.GET.get("type", "video")

        if upload_type == "image":
            return ImageCreateView.as_view()(request)
        else:
            return VideoCreateView.as_view()(request)


# ============ 헬퍼 함수 ============
def generate_thumbnail_from_video(video_path):
    """
    동영상 첫 프레임에서 썸네일 생성
    
    Method 1: ffmpeg-python 사용 (선호)
    Method 2: OpenCV 사용 (fallback)
    """
    print(f"\n--- 썸네일 생성 시작 ---")
    print(f"입력: {video_path}")
    
    # Method 1: ffmpeg-python 시도
    try:
        import ffmpeg
        
        print("🔧 방법 1: ffmpeg-python 사용")
        
        out, err = (
            ffmpeg.input(video_path, ss=0)
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        if out:
            print(f"✅ ffmpeg 출력: {len(out)} bytes")
            
            # PIL로 이미지 처리
            image = PILImage.open(BytesIO(out))
            print(f"🖼️ 원본 크기: {image.size}")
            
            # 리사이즈
            image.thumbnail((640, 360), PILImage.Resampling.LANCZOS)
            print(f"📏 리사이즈 후: {image.size}")
            
            # JPEG로 변환
            thumb_io = BytesIO()
            image.save(thumb_io, format="JPEG", quality=85)
            thumb_io.seek(0)
            
            print(f"💾 최종 크기: {len(thumb_io.getvalue())} bytes")
            print(f"✅ 썸네일 생성 성공 (ffmpeg)\n")
            
            return ContentFile(thumb_io.read())
        else:
            print("⚠️ ffmpeg 출력 없음")
            
    except ImportError:
        print("⚠️ ffmpeg-python 미설치")
    except Exception as e:
        print(f"⚠️ ffmpeg 오류: {e}")
    
    # Method 2: OpenCV 시도
    try:
        import cv2
        
        print("🔧 방법 2: OpenCV 사용")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ OpenCV로 동영상 열기 실패")
            return None
        
        # 첫 프레임 읽기
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("❌ 프레임 읽기 실패")
            return None
        
        print(f"🖼️ 프레임 크기: {frame.shape}")
        
        # BGR을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        image = PILImage.fromarray(frame_rgb)
        
        # 리사이즈
        image.thumbnail((640, 360), PILImage.Resampling.LANCZOS)
        print(f"📏 리사이즈 후: {image.size}")
        
        # JPEG로 저장
        thumb_io = BytesIO()
        image.save(thumb_io, format="JPEG", quality=85)
        thumb_io.seek(0)
        
        print(f"💾 최종 크기: {len(thumb_io.getvalue())} bytes")
        print(f"✅ 썸네일 생성 성공 (OpenCV)\n")
        
        return ContentFile(thumb_io.read())
        
    except ImportError:
        print("⚠️ OpenCV 미설치")
    except Exception as e:
        print(f"❌ OpenCV 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("❌ 썸네일 생성 실패 - 모든 방법 실패\n")
    return None