import os
import re
import ffmpeg
import mimetypes

from PIL import Image
from io import BytesIO

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
        self.object = form.save(commit=False)

        if self.object.file:
            self.object.file_size = self.object.file.size

        self.object.save()

        if self.object.file:
            thumbnail_content = generate_thumbnail(self.object.file.path)
            if thumbnail_content:
                original_name = os.path.splitext(
                    os.path.basename(self.object.file.name)
                )[0]
                thumbnail_name = f"{original_name}.jpg"
                self.object.thumbnail.save(
                    thumbnail_name, thumbnail_content, save=False
                )

        self.object.save()

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
        self.object = self.get_object()

        if self.object.file and os.path.isfile(self.object.file.path):
            os.remove(self.object.file.path)

        if self.object.thumbnail and os.path.isfile(self.object.thumbnail.path):
            os.remove(self.object.thumbnail.path)

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
                from PIL import Image as PILImage

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
        self.object = self.get_object()

        if self.object.file and os.path.isfile(self.object.file.path):
            os.remove(self.object.file.path)

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
def generate_thumbnail(video_path):
    try:
        out, _ = (
            ffmpeg.input(video_path, ss=0)
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True, capture_stderr=True)
        )

        image = Image.open(BytesIO(out))
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)

        thumb_io = BytesIO()
        image.save(thumb_io, format="JPEG", quality=85)
        thumb_io.seek(0)

        return ContentFile(thumb_io.read())
    except Exception as exc:
        print(f"썸네일 생성 오류: {exc}")
        return None
