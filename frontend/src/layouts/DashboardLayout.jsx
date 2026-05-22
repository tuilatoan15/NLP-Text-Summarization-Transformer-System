import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

const DashboardLayout = () => {
  return (
    <div className="flex h-screen bg-[#F8F9FA] dark:bg-slate-950 overflow-hidden text-gray-800 dark:text-slate-200 font-sans transition-colors duration-200">
      {/* Sidebar is fixed width */}
      <Sidebar />
      
      {/* Main content wrapper */}
      <div className="flex-1 flex flex-col ml-64 min-w-0">
        <Header />
        
        {/* Scrollable page content */}
        <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-7xl mx-auto w-full">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
