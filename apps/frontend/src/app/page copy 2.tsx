"use client";

/// 引入Tabs组件
// 引入Tabs组件
import { Tabs } from 'antd';
import React, { useEffect } from 'react';
import styles from './page.module.css';

const { TabPane } = Tabs;

export default function Page() {
  useEffect(() => {
    // 动态计算视口高度，并设置到CSS变量
    const setAppHeight = () => {
      const docEl = document.documentElement;
      docEl.style.setProperty('--app-height', `${window.innerHeight}px`);
    };
    window.addEventListener('resize', setAppHeight);
    // 初始化设置高度
    setAppHeight();
    
    // 清理监听器
    return () => window.removeEventListener('resize', setAppHeight);
  }, []);

  return (
    <div className={styles.fullHeight}>
      <header className={styles.header}>
        <h1>皮革纹理管理系统</h1>
      </header>

      <div className={styles.flexContent}>
        <Tabs tabPosition="left" className={styles.tabs}>
          <TabPane tab="纹理库" key="Library">
            {/* 纹理库内容 */}
            <h2>所有上传的纹理</h2>
          </TabPane>
          <TabPane tab="搜索纹理" key="Search">
            {/* 搜索纹理内容 */}
            <h2>搜索纹理</h2>
          </TabPane>
          <TabPane tab="新增纹理" key="Upload">
            {/* 新增纹理内容 */}
            <h2>新增纹理</h2>
            <div
              className={styles.uploadArea}
              onClick={() => {}}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {}}
            >
              拖拽图片到这里或点击进行拍照
              <input
                type="file"
                accept="image/*"
                className={styles.fileInput}
                onChange={() => {}}
              />
            </div>
          </TabPane>
        </Tabs>
      </div>

      <footer className={styles.footer}>
        <p>&copy; 2024 皮革纹理管理系统</p>
      </footer>
    </div>
  );
}
