"use client";

import React, { useState, useRef } from "react";
import { v4 as uuidv4 } from 'uuid';
import styles from "@/styles/TextureEdit.module.css"; // 假设你的样式文件名为TextureEdit.module.css
import { Button, Input, message } from "antd";

interface ImageInfo {
  src: string;
  file: File | null;
}

export function TextureEdit() {
  const [imageInfo, setImageInfo] = useState<ImageInfo>({
    src: "",
    file: null,
  });
  const [textureName, setTextureName] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const imgSrc = URL.createObjectURL(file);
      setImageInfo({ src: imgSrc, file });
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFiles({ target: { files } } as React.ChangeEvent<HTMLInputElement>);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const addCategory = async () => {
    if (!textureName.trim()) {
      message.error("请先输入纹理名称");
      return;
    }

    if (imageInfo.file) {
      const fileId = uuidv4();
      const formData = new FormData();
      formData.append("image", imageInfo.file);
      formData.append("name", textureName);
      formData.append("id", fileId);
      setUploading(true);
      try {
        const response = await fetch("http://localhost:4000/uploadImage", {
          method: "POST",
          body: formData,
        });
        setUploading(false);
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        console.log(data); // 处理响应数据
        // 重置表单
        setImageInfo({ src: "", file: null });
        setTextureName("");
        message.success("上传成功！");
      } catch (error) {
        setUploading(false);
        message.error("上传失败！");
        console.error("Upload failed", error);
      }
    }else {
      message.error("请选择一个文件");
    }
  };

  return (
    <div className={styles.editTexture}>
      <div
        className={styles.uploadArea}
        onClick={triggerFileInput}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{
          backgroundImage: imageInfo.src ? `url(${imageInfo.src})` : "",
        }}
      >
        {imageInfo.src ? "" : "拖拽图片到这里或点击进行拍照"}
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFiles}
        />
      </div>
      <div className={styles.editFooter}>
        <Input
          placeholder="请输入纹理名称"
          value={textureName}
          onChange={(e) => setTextureName(e.target.value)}
          style={{ marginTop: 16 }}
        />
        <Button type="primary" onClick={addCategory} style={{ marginTop: 16 }}>
          {uploading ? "上传中" : "提交"}
        </Button>
      </div>
    </div>
  );
}
