document.addEventListener('DOMContentLoaded', function() {
    // 平滑滚动到锚点
    document.querySelectorAll('nav ul li a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                // 计算header的实际高度，如果它是sticky的
                const header = document.querySelector('header');
                const headerHeight = header ? header.offsetHeight : 70; // 默认70px

                window.scrollTo({
                    top: targetElement.offsetTop - headerHeight,
                    behavior: 'smooth'
                });
            }
        });
    });

    // 导航链接高亮 (根据滚动位置)
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('main section[id]'); // 只选择main中的section

    function activateNavLink() {
        let currentSectionId = '';
        const headerHeight = document.querySelector('header') ? document.querySelector('header').offsetHeight : 70;
        const scrollPosition = pageYOffset + headerHeight + 20; // 增加20px的buffer

        sections.forEach(section => {
            if (section.offsetTop <= scrollPosition && (section.offsetTop + section.offsetHeight) > scrollPosition) {
                 currentSectionId = section.getAttribute('id');
            }
        });
        
        // 如果滚动到页面底部，且最后一个section不足够高以触发上面的逻辑，则激活最后一个nav link
        if ((window.innerHeight + window.pageYOffset) >= document.body.offsetHeight - 50) { // 50px buffer from bottom
            const lastSection = sections[sections.length - 1];
            if (lastSection) {
                 currentSectionId = lastSection.getAttribute('id');
            }
        }
        // 如果没有任何section匹配（比如滚动到最顶部，在hero之上），则取消所有active
        if (!currentSectionId && pageYOffset < sections[0].offsetTop - headerHeight) {
             navLinks.forEach(link => link.classList.remove('active'));
             // 如果需要首页高亮，可以单独处理
             const homeLink = document.querySelector('nav ul li a[href="#hero"]');
             if (homeLink && pageYOffset < sections[0].offsetTop - headerHeight) {
                homeLink.classList.add('active');
             }
             return;
        }


        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSectionId}`) {
                link.classList.add('active');
            }
        });
    }
    window.addEventListener('scroll', activateNavLink);
    activateNavLink(); // Initial call to set active link on page load

    // 页脚年份自动更新
    const currentYearElement = document.getElementById('current-year');
    if (currentYearElement) {
        currentYearElement.textContent = new Date().getFullYear();
    }

    console.log("米兰漫步网站脚本已更新并加载！");
});