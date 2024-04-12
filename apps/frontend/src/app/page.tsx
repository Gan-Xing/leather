"use client";

import { Tabs } from 'antd';
import React, { useEffect } from 'react';
import styles from './page.module.css';
import { TextureLibrary } from '@/components/TextureLibrary';
import { SearchTexture } from '@/components/SearchTexture';
import { TextureEdit } from '@/components/TextureEdit';
export default function Page() {
  const tabItems = [
    {
      label: '纹理库',
      key: 'Library',
      children: <TextureLibrary />,
    },
    {
      label: '搜索纹理',
      key: 'Search',
      children: <SearchTexture />,
    },
    {
      label: '新增纹理',
      key: 'Upload',
      children: <TextureEdit />,
    },
  ];

  return (
    <div className={styles.fullHeight}>
      <header className={styles.header}>
        皮革纹理管理系统
      </header>
      <main className={styles.main}>
        <Tabs defaultActiveKey="Library" tabPosition="bottom" type="card" items={tabItems} />
      </main>

    </div>
  );
}
